/**
 * WebGPU JavaScript Bindings
 *
 * This file exposes the WebGPU API to JavaScript via the JS engine abstraction.
 * Both Dawn and wgpu-native implement the same webgpu.h C API, so the bindings
 * work with either backend.
 *
 * Key APIs exposed:
 * - canvas (global) - represents the window
 * - canvas.getContext('webgpu') -> GPUCanvasContext
 * - navigator.gpu
 * - navigator.gpu.requestAdapter() -> GPUAdapter
 * - GPUAdapter.requestDevice() -> GPUDevice
 * - GPUDevice.createBuffer()
 * - GPUDevice.createShaderModule()
 * - GPUDevice.createRenderPipeline()
 * - GPUDevice.createCommandEncoder()
 * - GPUQueue.submit()
 */

#include "mystral/js/engine.h"
#include <iostream>
#include <vector>
#include <unordered_map>
#include <string>
#include <fstream>
#include <thread>
#include <chrono>
#include <mutex>
#include <deque>
#include <algorithm>
#include <cmath>
#include <array>
#include <sstream>
#include <unordered_set>
#include <atomic>
#if defined(__ANDROID__)
#include <android/log.h>
#include <unistd.h>
#endif
#include "mystral/webgpu/context.h"

// stb_image for image loading (implementation in stb_impl.cpp)
#include "stb_image.h"

// libwebp for WebP image decoding (optional - for GLTF EXT_texture_webp extension)
#ifdef MYSTRAL_HAS_WEBP
#include <webp/decode.h>
#endif

// GLTF/GLB loader
#include "mystral/gltf/gltf_loader.h"

// Canvas 2D context (Skia-backed)
#include "mystral/canvas/canvas2d.h"

// Forward declaration for Canvas2D bindings
namespace mystral {
namespace canvas {
    js::JSValueHandle createCanvas2DContext(js::Engine* engine, int width, int height);
}
}

// ============================================================================
// OffscreenCanvas - stores canvas element state for getContext support
// ============================================================================
struct OffscreenCanvas {
    int width = 300;
    int height = 150;
    mystral::js::JSValueHandle context2d;  // Cached 2D context (created on first getContext call)
    bool hasContext2d = false;
};

// Global storage for offscreen canvases (prevents them from being destroyed)
static std::unordered_map<int, std::unique_ptr<OffscreenCanvas>> g_offscreenCanvases;
static int g_nextOffscreenCanvasId = 0;

#if defined(MYSTRAL_WEBGPU_WGPU) || defined(MYSTRAL_WEBGPU_DAWN)
#include <webgpu/webgpu.h>
#include "mystral/webgpu_compat.h"
#endif

// wgpu-native specific extension functions (not in standard webgpu.h)
#if defined(MYSTRAL_WEBGPU_WGPU)
extern "C" {
// Device poll - blocks until GPU work is done
// From wgpu/wgpu.h but declared here to avoid include path issues
typedef struct WGPUWrappedSubmissionIndex WGPUWrappedSubmissionIndex;
WGPUBool wgpuDevicePoll(WGPUDevice device, WGPUBool wait, WGPUWrappedSubmissionIndex const* wrappedSubmissionIndex);
}
#endif

namespace mystral {
namespace webgpu {

// Verbose logging flag - controlled by --debug CLI flag or MYSTRAL_DEBUG=1
static bool g_verboseLogging = false;  // Disabled by default, enable with --debug

// Store references to WebGPU objects
static WGPUDevice g_device = nullptr;
static WGPUQueue g_queue = nullptr;
static WGPUSurface g_surface = nullptr;
static WGPUInstance g_instance = nullptr;
static js::Engine* g_engine = nullptr;
static WGPUAdapter g_adapter = nullptr;
static js::JSValueHandle g_jsDevice;
static bool g_deviceDestroyed = false;
static std::atomic<bool> g_deviceLost{false};
static std::atomic<uint64_t> g_outOfMemoryEvents{0};
static void* resourceData(js::JSValueHandle value);
static void cleanupOwnedResources();

struct DeferredResult {
    js::JSValueHandle promise, resolve, reject;
    bool settled = false;
};
struct CompilationMessage {
    std::string message, type;
    uint64_t lineNum, linePos, offset, length;
};
struct NativeCompletion {
    enum Kind { Error, Lost, Scope, Queue, Compilation } kind;
    std::shared_ptr<DeferredResult> deferred;
    bool success = true;
    int type = 0;
    std::string message;
    std::vector<CompilationMessage> compilation;
};
static std::mutex g_completionMutex;
static std::deque<NativeCompletion> g_completions;
static std::vector<std::weak_ptr<DeferredResult>> g_deferredResults;
static std::shared_ptr<DeferredResult> g_lostResult;

#if defined(MYSTRAL_WEBGPU_DAWN)
static std::string copyNativeString(WGPUStringView value) {
    if (!value.data) return {};
    return std::string(value.data, value.length == WGPU_STRLEN ? std::strlen(value.data) : value.length);
}
#else
static std::string copyNativeString(const char* value) { return value ? value : ""; }
#endif

static std::shared_ptr<DeferredResult> makeDeferred() {
    auto holder = g_engine->evalScriptWithResult(
        "(() => { let resolve, reject; const promise = new Promise((a,b) => {resolve=a; reject=b;});"
        "return {promise, resolve, reject}; })()", "<WebGPU promise>");
    auto result = std::make_shared<DeferredResult>();
    result->promise = g_engine->getProperty(holder, "promise");
    result->resolve = g_engine->getProperty(holder, "resolve");
    result->reject = g_engine->getProperty(holder, "reject");
    g_engine->protect(result->resolve);
    g_engine->protect(result->reject);
    g_deferredResults.push_back(result);
    return result;
}

static js::JSValueHandle nativeError(int type, const std::string& message) {
    auto error = g_engine->call(g_engine->getGlobalProperty("Error"), g_engine->newUndefined(),
                                {g_engine->newString(message.c_str())});
    const char* name = type == WGPUErrorType_Validation ? "GPUValidationError" :
        type == WGPUErrorType_OutOfMemory ? "GPUOutOfMemoryError" : "GPUInternalError";
    g_engine->setProperty(error, "name", g_engine->newString(name));
    return error;
}

static void settle(const std::shared_ptr<DeferredResult>& deferred, bool success, js::JSValueHandle value) {
    if (!g_engine || !deferred || deferred->settled) return;
    deferred->settled = true;
    g_engine->call(success ? deferred->resolve : deferred->reject, g_engine->newUndefined(), {value});
    g_engine->unprotect(deferred->resolve);
    g_engine->unprotect(deferred->reject);
}

static js::JSValueHandle rejected(const std::string& message) {
    auto deferred = makeDeferred();
    settle(deferred, false, nativeError(WGPUErrorType_Validation, message));
    return deferred->promise;
}

static js::JSValueHandle resolved(js::JSValueHandle value) {
    auto deferred = makeDeferred();
    settle(deferred, true, value);
    return deferred->promise;
}

static void enqueue(NativeCompletion completion) {
    std::lock_guard<std::mutex> lock(g_completionMutex);
    g_completions.push_back(std::move(completion));
}

void notifyUncapturedError(int type, const std::string& message) {
    if (type == WGPUErrorType_OutOfMemory) ++g_outOfMemoryEvents;
    NativeCompletion completion{NativeCompletion::Error};
    completion.type = type;
    completion.message = message;
    enqueue(std::move(completion));
}

void notifyDeviceLost(int reason, const std::string& message) {
    g_deviceLost.store(true);
    NativeCompletion completion{NativeCompletion::Lost};
    completion.type = reason;
    completion.message = message;
    enqueue(std::move(completion));
}

#if defined(MYSTRAL_WEBGPU_DAWN)
static void onScopeCompleted(WGPUPopErrorScopeStatus status, WGPUErrorType type, WGPUStringView message, void* userdata, void*) {
    std::unique_ptr<std::shared_ptr<DeferredResult>> owner(static_cast<std::shared_ptr<DeferredResult>*>(userdata));
    NativeCompletion completion{NativeCompletion::Scope};
    completion.deferred = *owner;
    completion.success = status == WGPUPopErrorScopeStatus_Success;
    completion.type = type;
    completion.message = copyNativeString(message);
    enqueue(std::move(completion));
}

static void onQueueCompleted(WGPUQueueWorkDoneStatus status, WGPUStringView message, void* userdata, void*) {
    std::unique_ptr<std::shared_ptr<DeferredResult>> owner(static_cast<std::shared_ptr<DeferredResult>*>(userdata));
    NativeCompletion completion{NativeCompletion::Queue};
    completion.deferred = *owner;
    completion.success = status == WGPUQueueWorkDoneStatus_Success;
    completion.message = copyNativeString(message);
    enqueue(std::move(completion));
}

static void onCompilationCompleted(WGPUCompilationInfoRequestStatus status, const WGPUCompilationInfo* info, void* userdata, void*) {
    std::unique_ptr<std::shared_ptr<DeferredResult>> owner(static_cast<std::shared_ptr<DeferredResult>*>(userdata));
    NativeCompletion completion{NativeCompletion::Compilation};
    completion.deferred = *owner;
    completion.success = status == WGPUCompilationInfoRequestStatus_Success && info;
    completion.message = "Native shader compilation information request failed";
    if (info) for (size_t i = 0; i < info->messageCount; ++i) {
        const auto& entry = info->messages[i];
        completion.compilation.push_back({copyNativeString(entry.message),
            entry.type == WGPUCompilationMessageType_Error ? "error" :
            entry.type == WGPUCompilationMessageType_Warning ? "warning" : "info",
            entry.lineNum, entry.linePos, entry.offset, entry.length});
    }
    enqueue(std::move(completion));
}
#endif

void processDeviceCallbacks() {
    if (!g_engine) return;
#if defined(MYSTRAL_WEBGPU_DAWN)
    if (g_device) wgpuDeviceTick(g_device);
    if (g_instance) wgpuInstanceProcessEvents(g_instance);
#elif defined(MYSTRAL_WEBGPU_WGPU)
    if (g_device) wgpuDevicePoll(g_device, false, nullptr);
#endif
    std::deque<NativeCompletion> pending;
    {
        std::lock_guard<std::mutex> lock(g_completionMutex);
        pending.swap(g_completions);
    }
    for (auto& completion : pending) {
        if (completion.kind == NativeCompletion::Error) {
            if (!g_jsDevice.ptr) continue;
            auto event = g_engine->evalScriptWithResult("new Event('uncapturederror', {cancelable:true})", "<WebGPU error event>");
            g_engine->setProperty(event, "error", nativeError(completion.type, completion.message));
            auto dispatch = g_engine->getProperty(g_jsDevice, "dispatchEvent");
            g_engine->call(dispatch, g_jsDevice, {event});
        } else if (completion.kind == NativeCompletion::Lost) {
            auto info = g_engine->newObject();
            g_engine->setProperty(info, "reason", g_engine->newString(
                completion.type == WGPUDeviceLostReason_Destroyed ? "destroyed" : "unknown"));
            g_engine->setProperty(info, "message", g_engine->newString(completion.message.c_str()));
            settle(g_lostResult, true, info);
        } else if (!completion.success) {
            settle(completion.deferred, false, nativeError(completion.type, completion.message));
        } else if (completion.kind == NativeCompletion::Scope) {
            settle(completion.deferred, true, completion.type == WGPUErrorType_NoError ? g_engine->newNull() :
                nativeError(completion.type, completion.message));
        } else if (completion.kind == NativeCompletion::Compilation) {
            auto info = g_engine->newObject();
            auto messages = g_engine->newArray(completion.compilation.size());
            for (size_t i = 0; i < completion.compilation.size(); ++i) {
                const auto& entry = completion.compilation[i];
                auto message = g_engine->newObject();
                g_engine->setProperty(message, "message", g_engine->newString(entry.message.c_str()));
                g_engine->setProperty(message, "type", g_engine->newString(entry.type.c_str()));
                g_engine->setProperty(message, "lineNum", g_engine->newNumber(entry.lineNum));
                g_engine->setProperty(message, "linePos", g_engine->newNumber(entry.linePos));
                g_engine->setProperty(message, "offset", g_engine->newNumber(entry.offset));
                g_engine->setProperty(message, "length", g_engine->newNumber(entry.length));
                g_engine->setPropertyIndex(messages, i, message);
            }
            g_engine->setProperty(info, "messages", messages);
            settle(completion.deferred, true, info);
        } else {
            settle(completion.deferred, true, g_engine->newUndefined());
        }
    }
    g_deferredResults.erase(std::remove_if(g_deferredResults.begin(), g_deferredResults.end(),
        [](const auto& value) { return value.expired(); }), g_deferredResults.end());
}

void cleanupDeviceBindings() {
    if (!g_engine) return;
    cleanupOwnedResources();
    for (auto& weak : g_deferredResults) if (auto deferred = weak.lock()) {
        if (!deferred->settled) {
            deferred->settled = true;
            g_engine->unprotect(deferred->resolve);
            g_engine->unprotect(deferred->reject);
        }
    }
    if (g_jsDevice.ptr) g_engine->unprotect(g_jsDevice);
    g_jsDevice = {};
    g_lostResult.reset();
    g_deferredResults.clear();
    g_engine = nullptr;
    std::lock_guard<std::mutex> lock(g_completionMutex);
    g_completions.clear();
}

static js::JSValueHandle nativeLimits(bool adapter) {
#if defined(MYSTRAL_WEBGPU_DAWN)
    WGPULimits limits = {};
    const auto status = adapter ? wgpuAdapterGetLimits(g_adapter, &limits) : wgpuDeviceGetLimits(g_device, &limits);
    if (status != WGPUStatus_Success) {
        g_engine->throwException("Native WebGPU limits query failed");
        return g_engine->newUndefined();
    }
#else
    WGPUSupportedLimits supported = {};
    if (!(adapter ? wgpuAdapterGetLimits(g_adapter, &supported) : wgpuDeviceGetLimits(g_device, &supported))) {
        g_engine->throwException("Native WebGPU limits query failed");
        return g_engine->newUndefined();
    }
    const auto& limits = supported.limits;
#endif
    auto object = g_engine->newObject();
#define EXPOSE_LIMIT(name) g_engine->setProperty(object, #name, g_engine->newNumber(static_cast<double>(limits.name)))
    EXPOSE_LIMIT(maxTextureDimension1D); EXPOSE_LIMIT(maxTextureDimension2D); EXPOSE_LIMIT(maxTextureDimension3D);
    EXPOSE_LIMIT(maxTextureArrayLayers); EXPOSE_LIMIT(maxBindGroups); EXPOSE_LIMIT(maxBindingsPerBindGroup);
    EXPOSE_LIMIT(maxDynamicUniformBuffersPerPipelineLayout); EXPOSE_LIMIT(maxDynamicStorageBuffersPerPipelineLayout);
    EXPOSE_LIMIT(maxSampledTexturesPerShaderStage); EXPOSE_LIMIT(maxSamplersPerShaderStage);
    EXPOSE_LIMIT(maxStorageBuffersPerShaderStage); EXPOSE_LIMIT(maxStorageTexturesPerShaderStage);
    EXPOSE_LIMIT(maxUniformBuffersPerShaderStage); EXPOSE_LIMIT(maxUniformBufferBindingSize);
    EXPOSE_LIMIT(maxStorageBufferBindingSize); EXPOSE_LIMIT(minUniformBufferOffsetAlignment);
    EXPOSE_LIMIT(minStorageBufferOffsetAlignment); EXPOSE_LIMIT(maxVertexBuffers); EXPOSE_LIMIT(maxBufferSize);
    EXPOSE_LIMIT(maxVertexAttributes); EXPOSE_LIMIT(maxVertexBufferArrayStride); EXPOSE_LIMIT(maxColorAttachments);
    EXPOSE_LIMIT(maxColorAttachmentBytesPerSample); EXPOSE_LIMIT(maxComputeWorkgroupStorageSize);
    EXPOSE_LIMIT(maxComputeInvocationsPerWorkgroup); EXPOSE_LIMIT(maxComputeWorkgroupSizeX);
    EXPOSE_LIMIT(maxComputeWorkgroupSizeY); EXPOSE_LIMIT(maxComputeWorkgroupSizeZ); EXPOSE_LIMIT(maxComputeWorkgroupsPerDimension);
#undef EXPOSE_LIMIT
    return object;
}

struct NamedFeature { const char* name; WGPUFeatureName native; };
static const NamedFeature kFeatures[] = {
    {"indirect-first-instance", WGPUFeatureName_IndirectFirstInstance},
    {"float32-filterable", WGPUFeatureName_Float32Filterable},
    {"texture-compression-bc", WGPUFeatureName_TextureCompressionBC},
    {"texture-compression-etc2", WGPUFeatureName_TextureCompressionETC2},
    {"texture-compression-astc", WGPUFeatureName_TextureCompressionASTC},
    {"depth-clip-control", WGPUFeatureName_DepthClipControl},
    {"depth32float-stencil8", WGPUFeatureName_Depth32FloatStencil8},
    {"timestamp-query", WGPUFeatureName_TimestampQuery},
    {"shader-f16", WGPUFeatureName_ShaderF16},
};

static js::JSValueHandle nativeFeatures(bool adapter) {
    auto set = g_engine->evalScriptWithResult("new Set()", "<WebGPU features>");
    auto add = g_engine->getProperty(set, "add");
    for (const auto& feature : kFeatures) {
        const bool enabled = adapter ? wgpuAdapterHasFeature(g_adapter, feature.native) : wgpuDeviceHasFeature(g_device, feature.native);
        if (enabled) g_engine->call(add, set, {g_engine->newString(feature.name)});
    }
    return set;
}

static std::string validateDeviceRequest(const std::vector<js::JSValueHandle>& args) {
    if (g_deviceDestroyed) return "Native GPUDevice was destroyed";
    if (args.empty()) return "";
    auto descriptor = args[0];
    auto requiredLimits = g_engine->getProperty(descriptor, "requiredLimits");
    if (!g_engine->isUndefined(requiredLimits)) {
        auto keysFunction = g_engine->getProperty(g_engine->getGlobalProperty("Object"), "keys");
        auto keys = g_engine->call(keysFunction, g_engine->newUndefined(), {requiredLimits});
        const auto count = static_cast<uint32_t>(g_engine->toNumber(g_engine->getProperty(keys, "length")));
        auto available = nativeLimits(false);
        for (uint32_t i = 0; i < count; ++i) {
            auto name = g_engine->toString(g_engine->getPropertyIndex(keys, i));
            auto supported = g_engine->getProperty(available, name.c_str());
            const double required = g_engine->toNumber(g_engine->getProperty(requiredLimits, name.c_str()));
            const double actual = g_engine->toNumber(supported);
            const bool alignment = name == "minUniformBufferOffsetAlignment" || name == "minStorageBufferOffsetAlignment";
            if (g_engine->isUndefined(supported) || !std::isfinite(required) || required < 0 || std::floor(required) != required ||
                (alignment ? actual > required : actual < required)) return "Unsupported required WebGPU limit: " + name;
        }
    }
    auto requiredFeatures = g_engine->getProperty(descriptor, "requiredFeatures");
    if (!g_engine->isUndefined(requiredFeatures)) {
        const auto count = static_cast<uint32_t>(g_engine->toNumber(g_engine->getProperty(requiredFeatures, "length")));
        for (uint32_t i = 0; i < count; ++i) {
            auto name = g_engine->toString(g_engine->getPropertyIndex(requiredFeatures, i));
            bool enabled = false;
            for (const auto& feature : kFeatures) if (name == feature.name)
                enabled = wgpuDeviceHasFeature(g_device, feature.native);
            if (!enabled) return "Required feature is not enabled on native device: " + name;
        }
    }
    return "";
}

static bool copyUnsigned(js::JSValueHandle value, uint64_t fallback, bool wide, uint64_t& output) {
    if (g_engine->isUndefined(value)) {
        output = fallback;
        return true;
    }
    const double number = std::trunc(g_engine->toNumber(value));
    if (!std::isfinite(number) || number < 0 || number >= (wide ? 18446744073709551616.0 : 4294967296.0)) {
        g_engine->throwException("WebGPU field is outside its unsigned integer range");
        return false;
    }
    output = static_cast<uint64_t>(number);
    return true;
}

static bool copyLayout(js::JSValueHandle descriptor, WGPUTextureDataLayout_Compat& layout) {
    uint64_t offset, bytesPerRow, rowsPerImage;
    if (!copyUnsigned(g_engine->getProperty(descriptor, "offset"), 0, true, offset) ||
        !copyUnsigned(g_engine->getProperty(descriptor, "bytesPerRow"), WGPU_COPY_STRIDE_UNDEFINED, false, bytesPerRow) ||
        !copyUnsigned(g_engine->getProperty(descriptor, "rowsPerImage"), WGPU_COPY_STRIDE_UNDEFINED, false, rowsPerImage)) return false;
    layout.offset = offset;
    layout.bytesPerRow = static_cast<uint32_t>(bytesPerRow);
    layout.rowsPerImage = static_cast<uint32_t>(rowsPerImage);
    return true;
}

static bool copyExtent(js::JSValueHandle value, WGPUExtent3D& extent) {
    const auto length = g_engine->getProperty(value, "length");
    const bool sequence = !g_engine->isUndefined(length);
    if (sequence) {
        const double count = g_engine->toNumber(length);
        if (count < 1 || count > 3) {
            g_engine->throwException("Texture copy extent sequence requires one to three coordinates");
            return false;
        }
    }
    auto width = sequence ? g_engine->getPropertyIndex(value, 0) : g_engine->getProperty(value, "width");
    if (g_engine->isUndefined(width)) {
        g_engine->throwException("Texture copy extent requires width");
        return false;
    }
    uint64_t w, h, d;
    if (!copyUnsigned(width, 0, false, w) ||
        !copyUnsigned(sequence ? g_engine->getPropertyIndex(value, 1) : g_engine->getProperty(value, "height"), 1, false, h) ||
        !copyUnsigned(sequence ? g_engine->getPropertyIndex(value, 2) : g_engine->getProperty(value, "depthOrArrayLayers"), 1, false, d)) return false;
    extent = {static_cast<uint32_t>(w), static_cast<uint32_t>(h), static_cast<uint32_t>(d)};
    return true;
}

static bool copyTexture(js::JSValueHandle value, WGPUImageCopyTexture_Compat& texture) {
    texture.texture = static_cast<WGPUTexture>(resourceData(g_engine->getProperty(value, "texture")));
    uint64_t mip;
    if (!copyUnsigned(g_engine->getProperty(value, "mipLevel"), 0, false, mip)) return false;
    texture.mipLevel = static_cast<uint32_t>(mip);
    texture.origin = {0, 0, 0};
    auto origin = g_engine->getProperty(value, "origin");
    if (!g_engine->isUndefined(origin)) {
        const bool sequence = !g_engine->isUndefined(g_engine->getProperty(origin, "length"));
        uint64_t x, y, z;
        if (!copyUnsigned(sequence ? g_engine->getPropertyIndex(origin, 0) : g_engine->getProperty(origin, "x"), 0, false, x) ||
            !copyUnsigned(sequence ? g_engine->getPropertyIndex(origin, 1) : g_engine->getProperty(origin, "y"), 0, false, y) ||
            !copyUnsigned(sequence ? g_engine->getPropertyIndex(origin, 2) : g_engine->getProperty(origin, "z"), 0, false, z)) return false;
        texture.origin = {static_cast<uint32_t>(x), static_cast<uint32_t>(y), static_cast<uint32_t>(z)};
    }
    texture.aspect = WGPUTextureAspect_All;
    auto aspect = g_engine->getProperty(value, "aspect");
    if (!g_engine->isUndefined(aspect)) {
        const auto name = g_engine->toString(aspect);
        if (name == "depth-only") texture.aspect = WGPUTextureAspect_DepthOnly;
        else if (name == "stencil-only") texture.aspect = WGPUTextureAspect_StencilOnly;
        else if (name != "all") {
            g_engine->throwException("Invalid texture copy aspect");
            return false;
        }
    }
    return true;
}

static bool sequenceArray(js::JSValueHandle value, js::JSValueHandle& array) {
    auto convert = g_engine->evalScriptWithResult(
        "(value => { if (value == null || typeof value[Symbol.iterator] !== 'function') "
        "throw new TypeError('WebGPU sequence requires an iterable'); return Array.from(value); })",
        "<WebGPU sequence conversion>");
    array = g_engine->call(convert, g_engine->newUndefined(), {value});
    // lastException is diagnostic history and survives a JavaScript catch.
    // Only the value produced by this conversion determines its success.
    return array.ptr && g_engine->isArray(array);
}

static bool dynamicOffsets(const std::vector<js::JSValueHandle>& args, std::vector<uint32_t>& offsets) {
    if (args.size() <= 2 || g_engine->isUndefined(args[2])) return true;
    js::JSValueHandle array;
    if (!sequenceArray(args[2], array)) return false;
    const auto count = static_cast<uint64_t>(g_engine->toNumber(g_engine->getProperty(array, "length")));
    uint64_t start = 0, length = count;
    if (args.size() > 3 && !copyUnsigned(args[3], 0, true, start)) return false;
    if (start > count) {
        g_engine->throwException("Dynamic offset start exceeds the provided array");
        return false;
    }
    length = count - start;
    if (args.size() > 4 && !copyUnsigned(args[4], length, false, length)) return false;
    if (length > count - start) {
        g_engine->throwException("Dynamic offset range exceeds the provided array");
        return false;
    }
    offsets.reserve(static_cast<size_t>(length));
    for (uint64_t i = 0; i < length; ++i) {
        uint64_t offset;
        if (!copyUnsigned(g_engine->getPropertyIndex(array, static_cast<uint32_t>(start + i)), 0, false, offset)) return false;
        offsets.push_back(static_cast<uint32_t>(offset));
    }
    return true;
}

static bool renderPassLoadOp(js::JSValueHandle value, WGPULoadOp& output) {
    output = WGPULoadOp_Undefined;
    if (g_engine->isUndefined(value)) return true;
    const auto name = g_engine->toString(value);
    if (name == "clear") output = WGPULoadOp_Clear;
    else if (name == "load") output = WGPULoadOp_Load;
    else { g_engine->throwException("Invalid render pass load operation"); return false; }
    return true;
}

static bool renderPassStoreOp(js::JSValueHandle value, WGPUStoreOp& output) {
    output = WGPUStoreOp_Undefined;
    if (g_engine->isUndefined(value)) return true;
    const auto name = g_engine->toString(value);
    if (name == "store") output = WGPUStoreOp_Store;
    else if (name == "discard") output = WGPUStoreOp_Discard;
    else { g_engine->throwException("Invalid render pass store operation"); return false; }
    return true;
}

// Offscreen rendering support (for no-SDL mode)
static WGPUTexture g_offscreenTexture = nullptr;
static WGPUTextureView g_offscreenTextureView = nullptr;

// Canvas context state
static WGPUTextureFormat g_surfaceFormat = WGPUTextureFormat_BGRA8UnormSrgb;  // Default, updated from context
static uint32_t g_canvasWidth = 800;
static uint32_t g_canvasHeight = 600;
static bool g_contextConfigured = false;
static WGPUTextureUsage g_surfaceUsage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_CopySrc;
static bool g_canvasFrameDiscarded = false;
#if defined(__ANDROID__)
static uint64_t g_discardedCanvasFrames = 0;
static int g_lastSurfaceAcquireStatus = 0;
static bool g_surfaceDiscardAwaitingRecovery = false;
static auto g_lastSurfaceDiscardLog = std::chrono::steady_clock::now();
static void logSurfaceDiscard(const char* action, bool force = false) {
    const auto now = std::chrono::steady_clock::now();
    if (!force && now - g_lastSurfaceDiscardLog < std::chrono::seconds(5)) return;
    g_lastSurfaceDiscardLog = now;
    __android_log_print(ANDROID_LOG_INFO, "MystralInterop",
        "SURFACE_ACQUIRE action=%s status=%d discardedFrames=%llu awaitingRecovery=%d",
        action, g_lastSurfaceAcquireStatus, static_cast<unsigned long long>(g_discardedCanvasFrames),
        g_surfaceDiscardAwaitingRecovery ? 1 : 0);
}
#endif

// Current frame's texture (refreshed each frame)
static WGPUTexture g_currentTexture = nullptr;
static WGPUTextureView g_currentTextureView = nullptr;
// Track the texture that g_currentTextureView was created from
// (needed for screenshot since g_currentTexture may change between createView and screenshot)
static WGPUTexture g_currentViewSourceTexture = nullptr;

// Screenshot support - persistent buffer for capturing frames
static WGPUBuffer g_screenshotBuffer = nullptr;
static size_t g_screenshotBufferSize = 0;
static uint32_t g_screenshotBytesPerRow = 0;

// Global state for render pass (needed for callbacks in lambdas)
// These need to be declared here so they're visible to all lambdas
static WGPURenderPassEncoder g_jsRenderPass = nullptr;
static WGPUComputePassEncoder g_jsComputePass = nullptr;
static WGPUCommandEncoder g_jsCommandEncoder = nullptr;

// Per-encoder render pass tracking (fixes issue with multiple encoders)
// Maps command encoder pointer to its active render pass encoder
static std::unordered_map<WGPUCommandEncoder, WGPURenderPassEncoder> g_encoderRenderPassMap;
static std::unordered_map<WGPUCommandEncoder, WGPUComputePassEncoder> g_encoderComputePassMap;

// Track whether the current frame's surface render pass has been ended
// This prevents presenting the surface before its render commands are submitted
static bool g_surfaceRenderPassEnded = false;
static WGPUCommandEncoder g_surfaceRenderEncoder = nullptr;
static bool g_screenshotPending = false;
static bool g_screenshotReady = false;
// Prevent capturing multiple screenshots per frame (Three.js does multiple queue.submit() per frame)
static bool g_screenshotCapturedThisFrame = false;
static std::vector<uint8_t> g_screenshotData;

// Main canvas 2D context (for Canvas 2D to WebGPU compositing)
static canvas::Canvas2DContext* g_mainCanvas2DContext = nullptr;

// Texture registry for tracking user-created textures
// Maps texture ID to {texture, format, dimensions, etc.}
struct TextureInfo {
    WGPUTexture texture;
    WGPUTextureFormat format;
    uint32_t width;
    uint32_t height;
    uint32_t depthOrArrayLayers;
    uint32_t mipLevelCount;
    WGPUTextureDimension dimension;  // 1D, 2D, or 3D
    uint64_t resourceId = 0;
};
static std::unordered_map<uint64_t, TextureInfo> g_textureRegistry;
static uint64_t g_nextTextureId = 1;

// Buffer registry for tracking buffers (needed for mapping operations)
struct BufferInfo {
    WGPUBuffer buffer;
    uint64_t size;
    WGPUBufferUsage usage;
    bool isMapped;
    void* mappedData;
    uint64_t mappedSize;
    WGPUMapMode mapMode;  // Track whether mapped for read or write
    uint64_t resourceId = 0;
};
static std::unordered_map<uint64_t, BufferInfo> g_bufferRegistry;
static uint64_t g_nextBufferId = 1;

// Pipeline registries for getBindGroupLayout support
static std::unordered_map<uint64_t, WGPUComputePipeline> g_computePipelineRegistry;
static uint64_t g_nextComputePipelineId = 1;
static std::unordered_map<uint64_t, WGPURenderPipeline> g_renderPipelineRegistry;
static uint64_t g_nextRenderPipelineId = 1;

// One ledger entry owns exactly one WebGPU API reference. JS wrappers carry a
// monotonic ID, so automatic canvas expiry/explicit destruction cannot leave a
// usable dangling native pointer. Weak callbacks and shutdown share the same
// idempotent release path; GPU command submissions keep Dawn's internal refs.
enum class ResourceKind {
    CanvasTexture, CanvasView, Buffer, Texture, TextureView, ShaderModule,
    RenderPipeline, ComputePipeline, Sampler, BindGroupLayout, BindGroup,
    PipelineLayout, CommandEncoder, CommandBuffer, RenderPass, ComputePass,
    RenderBundleEncoder, RenderBundle, Count
};
static const char* resourceNames[] = {
    "canvasTexture", "canvasView", "buffer", "texture", "textureView", "shaderModule",
    "renderPipeline", "computePipeline", "sampler", "bindGroupLayout", "bindGroup",
    "pipelineLayout", "commandEncoder", "commandBuffer", "renderPass", "computePass",
    "renderBundleEncoder", "renderBundle"
};
struct ResourceCount { uint64_t acquired = 0, released = 0, live = 0, peak = 0; };
struct OwnedResource {
    uint64_t id;
    ResourceKind kind;
    void* handle;
    uint64_t registryId;
    bool destroyed = false;
};
static std::unordered_map<uint64_t, OwnedResource> g_ownedResources;
// A private token outlives API-reference expiry until its actual wrapper GC.
// Public JavaScript properties are diagnostic only and cannot bypass expiry.
static std::unordered_map<void*, uint64_t> g_resourceTokens;
static uint64_t g_nextResourceId = 1;
static std::array<ResourceCount, static_cast<size_t>(ResourceKind::Count)> g_resourceCounts;
static ResourceCount g_resourceTotal;
static uint64_t g_wrapperCreated = 0, g_wrapperFinalized = 0, g_wrapperPeak = 0;
static uint64_t g_resourceSequence = 0, g_resourceFrame = 0;
static auto g_lastResourceLog = std::chrono::steady_clock::now();
static js::JSValueHandle g_canvasTextureObject;
static uint64_t g_canvasTextureResource = 0;
static std::vector<uint64_t> g_canvasViewResources;

static void logResourceSnapshot(const char* stage) {
    const auto now = std::chrono::steady_clock::now();
    if (std::string(stage) == "periodic" && now - g_lastResourceLog < std::chrono::seconds(5)) return;
    g_lastResourceLog = now;
    std::ostringstream text;
    auto count = [&](const ResourceCount& value) {
        text << "{\"acquired\":" << value.acquired << ",\"released\":" << value.released
             << ",\"live\":" << value.live << ",\"peak\":" << value.peak << "}";
    };
    text << "{\"schema\":\"mystral-webgpu-resource-snapshot-v1\",\"stage\":\"" << stage
         << "\",\"sequence\":" << ++g_resourceSequence << ",\"frame\":" << g_resourceFrame
         << ",\"steadyMs\":" << std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
#if defined(__ANDROID__)
    text << ",\"pid\":" << getpid();
#endif
    text << ",\"types\":{";
    for (size_t i = 0; i < g_resourceCounts.size(); ++i) {
        if (i) text << ',';
        text << '"' << resourceNames[i] << "\":";
        count(g_resourceCounts[i]);
    }
    text << "},\"total\":";
    count(g_resourceTotal);
    text << ",\"wrapper\":{\"created\":" << g_wrapperCreated << ",\"finalized\":" << g_wrapperFinalized
         << ",\"live\":" << g_wrapperCreated - g_wrapperFinalized << ",\"peak\":" << g_wrapperPeak << "}}";
#if defined(__ANDROID__)
    // Android's log formatting path truncates long messages. The consumer
    // reassembles these bounded, ordered ASCII slices before parsing JSON.
    const auto payload = text.str();
    constexpr size_t chunkBytes = 512;
    const size_t parts = (payload.size() + chunkBytes - 1) / chunkBytes;
    for (size_t index = 0; index < parts; ++index) {
        const auto chunk = payload.substr(index * chunkBytes, chunkBytes);
        __android_log_print(ANDROID_LOG_INFO, "MystralInterop", "RESOURCE_SNAPSHOT_PART %llu %zu/%zu %s",
            static_cast<unsigned long long>(g_resourceSequence), index, parts, chunk.c_str());
    }
#else
    std::cout << "RESOURCE_SNAPSHOT " << text.str() << std::endl;
#endif
}

static void releaseResource(uint64_t id, bool destroy = false) {
    auto it = g_ownedResources.find(id);
    if (it == g_ownedResources.end()) return;
    const auto resource = it->second;
    g_ownedResources.erase(it);
    if (resource.kind == ResourceKind::RenderPass) {
        for (auto owner = g_encoderRenderPassMap.begin(); owner != g_encoderRenderPassMap.end();) {
            if (owner->second == resource.handle) owner = g_encoderRenderPassMap.erase(owner);
            else ++owner;
        }
        if (g_jsRenderPass == resource.handle) g_jsRenderPass = nullptr;
    } else if (resource.kind == ResourceKind::ComputePass) {
        for (auto owner = g_encoderComputePassMap.begin(); owner != g_encoderComputePassMap.end();) {
            if (owner->second == resource.handle) owner = g_encoderComputePassMap.erase(owner);
            else ++owner;
        }
        if (g_jsComputePass == resource.handle) g_jsComputePass = nullptr;
    } else if (resource.kind == ResourceKind::CommandEncoder) {
        g_encoderRenderPassMap.erase(static_cast<WGPUCommandEncoder>(resource.handle));
        g_encoderComputePassMap.erase(static_cast<WGPUCommandEncoder>(resource.handle));
        if (g_jsCommandEncoder == resource.handle) g_jsCommandEncoder = nullptr;
    }
    if (resource.handle == g_currentTextureView) g_currentTextureView = nullptr;
    if (resource.registryId) {
        if (resource.kind == ResourceKind::Texture || resource.kind == ResourceKind::CanvasTexture)
            g_textureRegistry.erase(resource.registryId);
        else if (resource.kind == ResourceKind::Buffer) g_bufferRegistry.erase(resource.registryId);
        else if (resource.kind == ResourceKind::RenderPipeline) g_renderPipelineRegistry.erase(resource.registryId);
        else if (resource.kind == ResourceKind::ComputePipeline) g_computePipelineRegistry.erase(resource.registryId);
    }
    switch (resource.kind) {
    case ResourceKind::CanvasTexture:
    case ResourceKind::Texture:
        if (destroy && !resource.destroyed) wgpuTextureDestroy(static_cast<WGPUTexture>(resource.handle));
        wgpuTextureRelease(static_cast<WGPUTexture>(resource.handle)); break;
    case ResourceKind::CanvasView:
    case ResourceKind::TextureView: wgpuTextureViewRelease(static_cast<WGPUTextureView>(resource.handle)); break;
    case ResourceKind::Buffer:
        if (destroy) wgpuBufferDestroy(static_cast<WGPUBuffer>(resource.handle));
        wgpuBufferRelease(static_cast<WGPUBuffer>(resource.handle)); break;
    case ResourceKind::ShaderModule: wgpuShaderModuleRelease(static_cast<WGPUShaderModule>(resource.handle)); break;
    case ResourceKind::RenderPipeline: wgpuRenderPipelineRelease(static_cast<WGPURenderPipeline>(resource.handle)); break;
    case ResourceKind::ComputePipeline: wgpuComputePipelineRelease(static_cast<WGPUComputePipeline>(resource.handle)); break;
    case ResourceKind::Sampler: wgpuSamplerRelease(static_cast<WGPUSampler>(resource.handle)); break;
    case ResourceKind::BindGroupLayout: wgpuBindGroupLayoutRelease(static_cast<WGPUBindGroupLayout>(resource.handle)); break;
    case ResourceKind::BindGroup: wgpuBindGroupRelease(static_cast<WGPUBindGroup>(resource.handle)); break;
    case ResourceKind::PipelineLayout: wgpuPipelineLayoutRelease(static_cast<WGPUPipelineLayout>(resource.handle)); break;
    case ResourceKind::CommandEncoder: wgpuCommandEncoderRelease(static_cast<WGPUCommandEncoder>(resource.handle)); break;
    case ResourceKind::CommandBuffer: wgpuCommandBufferRelease(static_cast<WGPUCommandBuffer>(resource.handle)); break;
    case ResourceKind::RenderPass: wgpuRenderPassEncoderRelease(static_cast<WGPURenderPassEncoder>(resource.handle)); break;
    case ResourceKind::ComputePass: wgpuComputePassEncoderRelease(static_cast<WGPUComputePassEncoder>(resource.handle)); break;
    case ResourceKind::RenderBundleEncoder: wgpuRenderBundleEncoderRelease(static_cast<WGPURenderBundleEncoder>(resource.handle)); break;
    case ResourceKind::RenderBundle: wgpuRenderBundleRelease(static_cast<WGPURenderBundle>(resource.handle)); break;
    case ResourceKind::Count: break;
    }
    auto& counter = g_resourceCounts[static_cast<size_t>(resource.kind)];
    ++counter.released; --counter.live;
    ++g_resourceTotal.released; --g_resourceTotal.live;
}

static uint64_t manageResource(js::JSValueHandle object, void* handle, ResourceKind kind, uint64_t registryId = 0) {
    if (!handle) return 0;
    const uint64_t id = g_nextResourceId++;
    g_ownedResources.emplace(id, OwnedResource{id, kind, handle, registryId});
    auto* token = new uint64_t(id);
    g_resourceTokens.emplace(token, id);
    g_engine->setPrivateData(object, token);
    g_engine->setProperty(object, "_nativeResourceId", g_engine->newNumber(static_cast<double>(id)));
    if (registryId) {
        if (kind == ResourceKind::Texture || kind == ResourceKind::CanvasTexture) g_textureRegistry.at(registryId).resourceId = id;
        else if (kind == ResourceKind::Buffer) g_bufferRegistry.at(registryId).resourceId = id;
    }
    auto& counter = g_resourceCounts[static_cast<size_t>(kind)];
    ++counter.acquired; counter.peak = std::max(counter.peak, ++counter.live);
    ++g_resourceTotal.acquired; g_resourceTotal.peak = std::max(g_resourceTotal.peak, ++g_resourceTotal.live);
    ++g_wrapperCreated;
    g_wrapperPeak = std::max(g_wrapperPeak, g_wrapperCreated - g_wrapperFinalized);
    g_engine->registerRelease(object, [id, token]() {
        releaseResource(id);
        g_resourceTokens.erase(token);
        delete token;
        ++g_wrapperFinalized;
    });
    return id;
}

template <typename T> static T resourceHandle(uint64_t id) {
    const auto it = g_ownedResources.find(id);
    if (it == g_ownedResources.end() || it->second.destroyed) {
        if (g_engine) g_engine->throwException("WebGPU resource is expired, destroyed, or already consumed");
        return nullptr;
    }
    return static_cast<T>(it->second.handle);
}

void* resolveResourceData(void* data) {
    const auto token = g_resourceTokens.find(data);
    if (token != g_resourceTokens.end()) return resourceHandle<void*>(token->second);
    // The remaining data comes from Context-owned device/queue or Canvas2D,
    // never from a managed GPU wrapper (including after its API ref expires).
    return data;
}

static void* resourceData(js::JSValueHandle value) {
    if (g_engine->isUndefined(value) || g_engine->isNull(value)) return nullptr;
    return resolveResourceData(g_engine->getPrivateData(value));
}

static bool isCanvasView(WGPUTextureView view) {
    if (!view) return false;
    for (const auto id : g_canvasViewResources) {
        const auto resource = g_ownedResources.find(id);
        if (resource != g_ownedResources.end() && resource->second.handle == view) return true;
    }
    return false;
}

static void manageTextureView(js::JSValueHandle object, WGPUTextureView view, WGPUTexture texture) {
    const auto current = g_ownedResources.find(g_canvasTextureResource);
    const bool canvas = current != g_ownedResources.end() && current->second.handle == texture;
    const auto id = manageResource(object, view, canvas ? ResourceKind::CanvasView : ResourceKind::TextureView);
    if (canvas) g_canvasViewResources.push_back(id);
}

static void releaseReference(void* handle, ResourceKind kind) {
    for (const auto& entry : g_ownedResources) {
        if (entry.second.handle == handle && entry.second.kind == kind) {
            releaseResource(entry.first);
            return;
        }
    }
}

static void expireCanvasTexture() {
    for (const auto id : g_canvasViewResources) releaseResource(id);
    g_canvasViewResources.clear();
    // The lease owns only GetCurrentTexture's API reference. Surface still owns
    // its backing image, including when a frame acquired it but never presented.
    // Explicit texture.destroy() is separate from automatic wrapper expiry.
    if (g_canvasTextureResource) releaseResource(g_canvasTextureResource);
    g_canvasTextureResource = 0;
    if (g_canvasTextureObject.ptr && g_engine) g_engine->unprotect(g_canvasTextureObject);
    g_canvasTextureObject = {};
    g_currentTexture = nullptr;
    g_currentTextureView = nullptr;
    g_currentViewSourceTexture = nullptr;
    g_surfaceRenderEncoder = nullptr;
    g_surfaceRenderPassEnded = false;
    g_screenshotCapturedThisFrame = false;
    g_canvasFrameDiscarded = false;
}

static void cleanupInternalResources();
static void cleanupOwnedResources() {
    expireCanvasTexture();
    while (!g_ownedResources.empty()) releaseResource(g_ownedResources.begin()->first);
    g_encoderRenderPassMap.clear(); g_encoderComputePassMap.clear();
    g_jsRenderPass = nullptr; g_jsComputePass = nullptr; g_jsCommandEncoder = nullptr;
    cleanupInternalResources();
#if defined(__ANDROID__)
    if (g_discardedCanvasFrames) logSurfaceDiscard("cleanup", true);
#endif
    logResourceSnapshot("cleanup");
}

// Dawn resource cleanup is handled via Engine::registerRelease(), which sets up
// V8 weak callbacks. When the JS wrapper object is garbage collected (no more
// JS references), the callback fires and releases the Dawn resource.
// This is the same pattern Chrome uses for WebGPU resource lifecycle.

// Buffer map callback data (global for static callback)
struct BufferMapData {
    bool completed = false;
    WGPUBufferMapAsyncStatus_Compat status = WGPUBufferMapAsyncStatus_Unknown_Compat;
    std::string errorMessage;
};
static BufferMapData g_bufferMapData;

#if WGPU_BUFFER_MAP_USES_CALLBACK_INFO
// Dawn buffer map callback (4 params)
static void onBufferMapped(WGPUMapAsyncStatus status, WGPUStringView message, void* userdata1, void* userdata2) {
    auto* data = (BufferMapData*)userdata1;
    data->status = status;
    data->completed = true;
    if (message.data && message.length > 0) {
        data->errorMessage = std::string(message.data, message.length);
    }
}
#else
// wgpu-native buffer map callback (2 params)
static void onBufferMapped(WGPUBufferMapAsyncStatus status, void* userdata) {
    auto* data = (BufferMapData*)userdata;
    data->status = status;
    data->completed = true;
}
#endif

/**
 * Convert texture format enum to string
 */
static const char* formatToString(WGPUTextureFormat format) {
    switch (format) {
        case WGPUTextureFormat_BGRA8Unorm: return "bgra8unorm";
        case WGPUTextureFormat_BGRA8UnormSrgb: return "bgra8unorm-srgb";
        case WGPUTextureFormat_RGBA8Unorm: return "rgba8unorm";
        case WGPUTextureFormat_RGBA8UnormSrgb: return "rgba8unorm-srgb";
        case WGPUTextureFormat_R8Unorm: return "r8unorm";
        case WGPUTextureFormat_RG8Unorm: return "rg8unorm";
        case WGPUTextureFormat_R16Float: return "r16float";
        case WGPUTextureFormat_RG16Float: return "rg16float";
        case WGPUTextureFormat_R32Float: return "r32float";
        case WGPUTextureFormat_RG32Float: return "rg32float";
        case WGPUTextureFormat_RGBA16Float: return "rgba16float";
        case WGPUTextureFormat_RGBA32Float: return "rgba32float";
        case WGPUTextureFormat_Depth24Plus: return "depth24plus";
        case WGPUTextureFormat_Depth24PlusStencil8: return "depth24plus-stencil8";
        case WGPUTextureFormat_Depth32Float: return "depth32float";
        default: return "bgra8unorm";  // Default
    }
}

/**
 * Parse texture format string to enum
 */
static WGPUTextureFormat stringToFormat(const std::string& format) {
    if (format == "bgra8unorm") return WGPUTextureFormat_BGRA8Unorm;
    if (format == "bgra8unorm-srgb") return WGPUTextureFormat_BGRA8UnormSrgb;
    if (format == "rgba8unorm") return WGPUTextureFormat_RGBA8Unorm;
    if (format == "rgba8unorm-srgb") return WGPUTextureFormat_RGBA8UnormSrgb;
    if (format == "r8unorm") return WGPUTextureFormat_R8Unorm;
    if (format == "rg8unorm") return WGPUTextureFormat_RG8Unorm;
    if (format == "r16float") return WGPUTextureFormat_R16Float;
    if (format == "rg16float") return WGPUTextureFormat_RG16Float;
    if (format == "r32float") return WGPUTextureFormat_R32Float;
    if (format == "rg32float") return WGPUTextureFormat_RG32Float;
    if (format == "rgba16float") return WGPUTextureFormat_RGBA16Float;
    if (format == "rgba32float") return WGPUTextureFormat_RGBA32Float;
    if (format == "depth24plus") return WGPUTextureFormat_Depth24Plus;
    if (format == "depth24plus-stencil8") return WGPUTextureFormat_Depth24PlusStencil8;
    if (format == "depth32float") return WGPUTextureFormat_Depth32Float;
    // Log unrecognized formats for debugging
    if (!format.empty()) {
        std::cerr << "[WebGPU] Warning: Unrecognized format '" << format << "', defaulting to BGRA8Unorm" << std::endl;
    }
    return WGPUTextureFormat_BGRA8Unorm;  // Default to non-sRGB
}

/**
 * Parse texture dimension string to enum
 */
static WGPUTextureDimension stringToTextureDimension(const std::string& dim) {
    if (dim == "1d") return WGPUTextureDimension_1D;
    if (dim == "2d") return WGPUTextureDimension_2D;
    if (dim == "3d") return WGPUTextureDimension_3D;
    return WGPUTextureDimension_2D;  // Default
}

/**
 * Parse texture view dimension string to enum
 */
static WGPUTextureViewDimension stringToTextureViewDimension(const std::string& dim) {
    if (dim == "1d") return WGPUTextureViewDimension_1D;
    if (dim == "2d") return WGPUTextureViewDimension_2D;
    if (dim == "2d-array") return WGPUTextureViewDimension_2DArray;
    if (dim == "cube") return WGPUTextureViewDimension_Cube;
    if (dim == "cube-array") return WGPUTextureViewDimension_CubeArray;
    if (dim == "3d") return WGPUTextureViewDimension_3D;
    return WGPUTextureViewDimension_2D;  // Default
}

/**
 * Parse address mode string to enum
 */
static WGPUAddressMode stringToAddressMode(const std::string& mode) {
    if (mode == "clamp-to-edge") return WGPUAddressMode_ClampToEdge;
    if (mode == "repeat") return WGPUAddressMode_Repeat;
    if (mode == "mirror-repeat") return WGPUAddressMode_MirrorRepeat;
    return WGPUAddressMode_ClampToEdge;  // Default
}

/**
 * Parse filter mode string to enum
 */
static WGPUFilterMode stringToFilterMode(const std::string& mode) {
    if (mode == "nearest") return WGPUFilterMode_Nearest;
    if (mode == "linear") return WGPUFilterMode_Linear;
    return WGPUFilterMode_Nearest;  // Default
}

/**
 * Parse mipmap filter mode string to enum
 */
static WGPUMipmapFilterMode stringToMipmapFilterMode(const std::string& mode) {
    if (mode == "nearest") return WGPUMipmapFilterMode_Nearest;
    if (mode == "linear") return WGPUMipmapFilterMode_Linear;
    return WGPUMipmapFilterMode_Nearest;  // Default
}

/**
 * Parse compare function string to enum
 */
static WGPUCompareFunction stringToCompareFunction(const std::string& func) {
    if (func == "never") return WGPUCompareFunction_Never;
    if (func == "less") return WGPUCompareFunction_Less;
    if (func == "equal") return WGPUCompareFunction_Equal;
    if (func == "less-equal") return WGPUCompareFunction_LessEqual;
    if (func == "greater") return WGPUCompareFunction_Greater;
    if (func == "not-equal") return WGPUCompareFunction_NotEqual;
    if (func == "greater-equal") return WGPUCompareFunction_GreaterEqual;
    if (func == "always") return WGPUCompareFunction_Always;
    return WGPUCompareFunction_Undefined;  // Default (no comparison)
}

/**
 * Get the current swapchain texture (or offscreen texture in no-SDL mode)
 */
static WGPUTexture getCurrentSwapchainTexture() {
#if defined(MYSTRAL_WEBGPU_WGPU) || defined(MYSTRAL_WEBGPU_DAWN)
    // In no-SDL mode, use the offscreen texture
    if (!g_surface) {
        if (g_offscreenTexture) {
            return g_offscreenTexture;
        }
        std::cerr << "[WebGPU] No surface and no offscreen texture available" << std::endl;
        return nullptr;
    }

    WGPUSurfaceTexture surfaceTexture = {};
    wgpuSurfaceGetCurrentTexture(g_surface, &surfaceTexture);

#if defined(__ANDROID__)
    g_lastSurfaceAcquireStatus = static_cast<int>(surfaceTexture.status);
    const bool acquired = wgpuSurfaceTextureStatusIsSuccess(surfaceTexture.status) ||
        surfaceTexture.status == WGPUSurfaceGetCurrentTextureStatus_SuccessSuboptimal_Compat;
#else
    const bool acquired = wgpuSurfaceTextureStatusIsSuccess(surfaceTexture.status);
#endif
    if (!acquired) {
        if (surfaceTexture.texture) wgpuTextureRelease(surfaceTexture.texture);
#if defined(__ANDROID__)
        const bool transient = surfaceTexture.status == WGPUSurfaceGetCurrentTextureStatus_Timeout ||
            surfaceTexture.status == WGPUSurfaceGetCurrentTextureStatus_Outdated ||
            surfaceTexture.status == WGPUSurfaceGetCurrentTextureStatus_Lost;
        if (transient && g_contextConfigured && g_device && !g_deviceDestroyed &&
                !g_deviceLost.load() && g_outOfMemoryEvents.load() == 0 && g_canvasWidth && g_canvasHeight) {
            // A Surface can disappear after owner-thread preparation, while its
            // JavaScript frame is already executing. Finish that one frame on a
            // disposable texture rather than throwing away the application's RAF.
            // Only known transient statuses qualify; Error, OOM and device loss
            // remain visible failures and can never become a presentation PASS.
#if defined(MYSTRAL_WEBGPU_DAWN)
            wgpuDeviceTick(g_device);
            if (g_instance) wgpuInstanceProcessEvents(g_instance);
#elif defined(MYSTRAL_WEBGPU_WGPU)
            wgpuDevicePoll(g_device, false, nullptr);
#endif
            if (!g_deviceLost.load() && g_outOfMemoryEvents.load() == 0) {
                WGPUTextureDescriptor descriptor = {};
                descriptor.size = {g_canvasWidth, g_canvasHeight, 1};
                descriptor.format = g_surfaceFormat;
                descriptor.usage = g_surfaceUsage;
                descriptor.dimension = WGPUTextureDimension_2D;
                descriptor.mipLevelCount = 1;
                descriptor.sampleCount = 1;
                auto discarded = wgpuDeviceCreateTexture(g_device, &descriptor);
                if (discarded && !g_deviceLost.load() && g_outOfMemoryEvents.load() == 0) {
                    const bool first = !g_surfaceDiscardAwaitingRecovery;
                    g_canvasFrameDiscarded = true;
                    ++g_discardedCanvasFrames;
                    g_surfaceDiscardAwaitingRecovery = true;
                    logSurfaceDiscard("discard", first);
                    return discarded; // This API reference enters the ordinary canvas lease ledger.
                }
                if (discarded) wgpuTextureRelease(discarded);
            }
        }
        __android_log_print(ANDROID_LOG_ERROR, "MystralInterop",
            "SURFACE_ACQUIRE_FATAL status=%d deviceDestroyed=%d deviceLost=%d oomEvents=%llu configured=%d",
            g_lastSurfaceAcquireStatus, g_deviceDestroyed ? 1 : 0, g_deviceLost.load() ? 1 : 0,
            static_cast<unsigned long long>(g_outOfMemoryEvents.load()), g_contextConfigured ? 1 : 0);
#endif
        std::cerr << "[WebGPU] Failed to get current texture" << std::endl;
        return nullptr;
    }

#if defined(__ANDROID__)
    if (g_surfaceDiscardAwaitingRecovery) {
        g_surfaceDiscardAwaitingRecovery = false;
        logSurfaceDiscard("recovered", true);
    }
#endif
    return surfaceTexture.texture;
#else
    return nullptr;
#endif
}

static js::JSValueHandle acquireCanvasTexture() {
    if (g_deviceDestroyed || !g_contextConfigured) {
        g_engine->throwException("WebGPU canvas context is not configured with a live device");
        return g_engine->newUndefined();
    }
    // getCurrentTexture returns the same object until automatic expiry, including
    // after an explicit destroy(). It must not acquire another API ref per call.
    if (g_canvasTextureObject.ptr) return g_canvasTextureObject;
    auto texture = getCurrentSwapchainTexture();
    if (!texture) {
        g_engine->throwException("Failed to get current texture");
        return g_engine->newUndefined();
    }
    if (!g_surface) {
        // The headless texture is borrowed from Context; obtain our own lease ref.
#if defined(MYSTRAL_WEBGPU_DAWN)
        wgpuTextureAddRef(texture);
#else
        wgpuTextureReference(texture);
#endif
    }
    const uint64_t textureId = g_nextTextureId++;
    g_textureRegistry[textureId] = {texture, g_surfaceFormat, g_canvasWidth, g_canvasHeight, 1, 1, WGPUTextureDimension_2D};
    auto object = g_engine->newObject();
    const uint64_t resourceId = manageResource(object, texture, ResourceKind::CanvasTexture, textureId);
    g_canvasTextureObject = object;
    g_canvasTextureResource = resourceId;
    g_engine->protect(object);
    g_currentTexture = texture;
    g_engine->setProperty(object, "width", g_engine->newNumber(g_canvasWidth));
    g_engine->setProperty(object, "height", g_engine->newNumber(g_canvasHeight));
    g_engine->setProperty(object, "depthOrArrayLayers", g_engine->newNumber(1));
    g_engine->setProperty(object, "format", g_engine->newString(formatToString(g_surfaceFormat)));
    g_engine->setProperty(object, "_textureId", g_engine->newNumber(static_cast<double>(textureId)));
    g_engine->setProperty(object, "createView", g_engine->newFunction("createView",
        [resourceId](void*, const std::vector<js::JSValueHandle>&) {
            auto texture = resourceHandle<WGPUTexture>(resourceId);
            if (!texture) return g_engine->newUndefined();
            WGPUTextureViewDescriptor descriptor = {};
            descriptor.format = g_surfaceFormat;
            descriptor.dimension = WGPUTextureViewDimension_2D;
            descriptor.mipLevelCount = 1;
            descriptor.arrayLayerCount = 1;
            descriptor.aspect = WGPUTextureAspect_All;
            auto view = wgpuTextureCreateView(texture, &descriptor);
            if (!view) {
                g_engine->throwException("Failed to create canvas texture view");
                return g_engine->newUndefined();
            }
            auto object = g_engine->newObject();
            const auto id = manageResource(object, view, ResourceKind::CanvasView);
            g_canvasViewResources.push_back(id);
            g_currentTextureView = view;
            g_currentViewSourceTexture = texture;
            g_engine->setProperty(object, "_type", g_engine->newString("textureView"));
            return object;
        }));
    g_engine->setProperty(object, "destroy", g_engine->newFunction("destroy",
        [resourceId](void*, const std::vector<js::JSValueHandle>&) {
            const auto it = g_ownedResources.find(resourceId);
            if (it != g_ownedResources.end() && !it->second.destroyed) {
                // Destroy this JS canvas lease, not Surface's backing image.
                // Existing views become unusable, while getCurrentTexture keeps
                // returning this same destroyed wrapper until frame expiry.
                it->second.destroyed = true;
                for (const auto id : g_canvasViewResources) releaseResource(id);
                g_canvasViewResources.clear();
                g_currentTextureView = nullptr;
                g_currentViewSourceTexture = nullptr;
            }
            return g_engine->newUndefined();
        }));
    return object;
}

void suspendSurfaceBindings(WGPUSurface surface) {
    if (g_surface != surface) return;
    expireCanvasTexture();
    g_surface = nullptr;
    // The texture registry owns the texture's API reference; these are only frame aliases.
    g_currentTexture = nullptr;
    g_currentViewSourceTexture = nullptr;
    g_surfaceRenderEncoder = nullptr;
    g_surfaceRenderPassEnded = false;
}

void retireSurfaceBindings(WGPUSurface surface) {
    if (g_surface != surface) return;
    suspendSurfaceBindings(surface);
    g_contextConfigured = false;
}

bool restoreSurfaceBindings(WGPUSurface surface) {
    if (!surface || !g_device || g_deviceDestroyed) return false;
    g_surface = surface;
    if (g_contextConfigured) {
        WGPUSurfaceConfiguration configuration = {};
        configuration.device = g_device;
        configuration.format = g_surfaceFormat;
        configuration.width = g_canvasWidth;
        configuration.height = g_canvasHeight;
        configuration.usage = g_surfaceUsage;
        configuration.alphaMode = WGPUCompositeAlphaMode_Auto;
        configuration.presentMode = WGPUPresentMode_Fifo;
        wgpuSurfaceConfigure(g_surface, &configuration);
    }
    return true;
}

/**
 * Initialize WebGPU bindings in the JS engine
 */
bool initBindings(js::Engine* engine, void* wgpuInstance, void* wgpuDevice, void* wgpuQueue, void* wgpuSurface, uint32_t surfaceFormat, uint32_t width, uint32_t height, bool debug, void* wgpuAdapter) {
    if (!engine) {
        std::cerr << "[WebGPU] No JS engine provided for bindings" << std::endl;
        return false;
    }

#if defined(MYSTRAL_WEBGPU_WGPU) || defined(MYSTRAL_WEBGPU_DAWN)
    // Set verbose logging based on debug flag
    g_verboseLogging = debug;

    g_engine = engine;
    g_instance = (WGPUInstance)wgpuInstance;
    g_device = (WGPUDevice)wgpuDevice;
    g_queue = (WGPUQueue)wgpuQueue;
    g_surface = (WGPUSurface)wgpuSurface;
    g_adapter = (WGPUAdapter)wgpuAdapter;
    g_deviceDestroyed = false;
    g_deviceLost.store(false);
    g_outOfMemoryEvents.store(0);
    g_contextConfigured = false;
    g_surfaceUsage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_CopySrc;

    // Set canvas dimensions from window size
    g_canvasWidth = width;
    g_canvasHeight = height;
    g_surfaceFormat = (WGPUTextureFormat)surfaceFormat;
    logResourceSnapshot("init");

    if (g_verboseLogging) {
        std::cout << "[WebGPU] Initializing JavaScript bindings..." << std::endl;
        std::cout << "[WebGPU] Surface format: " << surfaceFormat << std::endl;
    }

    // ========================================================================
    // Create a mock parent element for the canvas (needed by Debugger)
    // ========================================================================
    auto parentElement = engine->newObject();
    engine->setProperty(parentElement, "style", engine->newObject());
    engine->setProperty(parentElement, "appendChild",
        engine->newFunction("appendChild", [](void* ctx, const std::vector<js::JSValueHandle>& args) {
            // No-op in native runtime
            return args.empty() ? g_engine->newUndefined() : args[0];
        })
    );
    engine->setProperty(parentElement, "removeChild",
        engine->newFunction("removeChild", [](void* ctx, const std::vector<js::JSValueHandle>& args) {
            return args.empty() ? g_engine->newUndefined() : args[0];
        })
    );

    // ========================================================================
    // Get existing canvas from runtime.cpp's document.getElementById
    // The canvas was created by setupDOMEvents() with addEventListener, style, etc.
    // We just need to add WebGPU-specific methods (getContext) to it.
    // ========================================================================
    auto existingDocument = engine->getGlobalProperty("document");
    auto getElementByIdFunc = engine->getProperty(existingDocument, "getElementById");

    // Call document.getElementById('canvas') to get the existing canvas
    std::vector<js::JSValueHandle> args;
    args.push_back(engine->newString("canvas"));
    auto canvasObject = engine->call(getElementByIdFunc, existingDocument, args);

    if (engine->isNull(canvasObject) || engine->isUndefined(canvasObject)) {
        std::cerr << "[WebGPU] Warning: No existing canvas found, creating new one" << std::endl;
        canvasObject = engine->newObject();
        engine->setProperty(canvasObject, "width", engine->newNumber(g_canvasWidth));
        engine->setProperty(canvasObject, "height", engine->newNumber(g_canvasHeight));
        engine->setProperty(canvasObject, "clientWidth", engine->newNumber(g_canvasWidth));
        engine->setProperty(canvasObject, "clientHeight", engine->newNumber(g_canvasHeight));
    }

    // Update canvas dimensions (in case they differ)
    engine->setProperty(canvasObject, "width", engine->newNumber(g_canvasWidth));
    engine->setProperty(canvasObject, "height", engine->newNumber(g_canvasHeight));
    engine->setProperty(canvasObject, "clientWidth", engine->newNumber(g_canvasWidth));
    engine->setProperty(canvasObject, "clientHeight", engine->newNumber(g_canvasHeight));

    // canvas.parentElement - mock parent element (for Debugger compatibility)
    engine->setProperty(canvasObject, "parentElement", parentElement);

    // canvas.getContext('webgpu') -> GPUCanvasContext
    // This is the WebGPU-specific method we add to the existing canvas
    engine->setProperty(canvasObject, "getContext",
        engine->newFunction("getContext", [](void* ctx, const std::vector<js::JSValueHandle>& args) {
            if (args.empty()) {
                return g_engine->newNull();
            }

            std::string contextType = g_engine->toString(args[0]);

            // Handle Canvas 2D context
            if (contextType == "2d") {
                if (g_verboseLogging) std::cout << "[Canvas] Creating 2D context (" << g_canvasWidth << "x" << g_canvasHeight << ")" << std::endl;
                auto ctx2d = canvas::createCanvas2DContext(g_engine, g_canvasWidth, g_canvasHeight);

                // Set reference back to canvas
                auto canvas = g_engine->getGlobalProperty("canvas");
                g_engine->setProperty(ctx2d, "canvas", canvas);

                // Store the native context for Canvas 2D to WebGPU compositing
                g_mainCanvas2DContext = static_cast<canvas::Canvas2DContext*>(resourceData(ctx2d));
                if (g_verboseLogging) std::cout << "[Canvas] Main canvas using 2D context - will composite to WebGPU" << std::endl;

                return ctx2d;
            }

            if (contextType != "webgpu") {
                std::cerr << "[Canvas] Unknown context type: " << contextType << std::endl;
                return g_engine->newNull();
            }

            // Create GPUCanvasContext
            auto canvasContext = g_engine->newObject();

            // Methods consult the live Context-owned surface, never a stale borrowed private pointer.

            // context.canvas - reference back to canvas
            auto canvas = g_engine->getGlobalProperty("canvas");
            g_engine->setProperty(canvasContext, "canvas", canvas);

            // context.configure({ device, format, alphaMode })
            g_engine->setProperty(canvasContext, "configure",
                g_engine->newFunction("configure", [](void* ctx, const std::vector<js::JSValueHandle>& args) {
                    if (g_deviceDestroyed) {
                        g_engine->throwException("The single-device Runtime is destroyed; create a new Runtime");
                        return g_engine->newUndefined();
                    }
                    if (args.empty()) {
                        g_engine->throwException("configure requires a descriptor");
                        return g_engine->newUndefined();
                    }

                    auto descriptor = args[0];

                    // Get format
                    std::string format = g_engine->toString(g_engine->getProperty(descriptor, "format"));
                    g_surfaceFormat = stringToFormat(format);
                    auto canvas = g_engine->getGlobalProperty("canvas");
                    auto width = g_engine->toNumber(g_engine->getProperty(canvas, "width"));
                    auto height = g_engine->toNumber(g_engine->getProperty(canvas, "height"));
                    auto configuredDevice = g_engine->getProperty(descriptor, "device");
                    if (!std::isfinite(width) || !std::isfinite(height) || width <= 0 || height <= 0 ||
                        resourceData(configuredDevice) != g_device) {
                        g_engine->throwException("Invalid WebGPU canvas dimensions or device");
                        return g_engine->newUndefined();
                    }
                    g_canvasWidth = static_cast<uint32_t>(width);
                    g_canvasHeight = static_cast<uint32_t>(height);
                    auto usage = g_engine->getProperty(descriptor, "usage");
                    g_surfaceUsage = (g_engine->isUndefined(usage) ? WGPUTextureUsage_RenderAttachment :
                        static_cast<WGPUTextureUsage>(static_cast<uint64_t>(g_engine->toNumber(usage)))) |
                        WGPUTextureUsage_CopySrc;
                    if (g_surface) {
                        WGPUSurfaceConfiguration configuration = {};
                        configuration.device = g_device;
                        configuration.format = g_surfaceFormat;
                        configuration.width = g_canvasWidth;
                        configuration.height = g_canvasHeight;
                        // Retain native screenshot support without confusing it with external presentation evidence.
                        configuration.usage = g_surfaceUsage;
                        configuration.alphaMode = WGPUCompositeAlphaMode_Auto;
                        configuration.presentMode = WGPUPresentMode_Fifo;
                        wgpuSurfaceConfigure(g_surface, &configuration);
                    }

                    g_contextConfigured = true;
                    if (g_verboseLogging) std::cout << "[Canvas] Context configured with format: " << format << std::endl;

                    return g_engine->newUndefined();
                })
            );

            // context.unconfigure()
            g_engine->setProperty(canvasContext, "unconfigure",
                g_engine->newFunction("unconfigure", [](void* ctx, const std::vector<js::JSValueHandle>& args) {
                    g_contextConfigured = false;
                    if (g_surface) wgpuSurfaceUnconfigure(g_surface);
                    return g_engine->newUndefined();
                })
            );

            // context.getCurrentTexture() -> GPUTexture
            g_engine->setProperty(canvasContext, "getCurrentTexture",
                g_engine->newFunction("getCurrentTexture", [](void* ctx, const std::vector<js::JSValueHandle>& args) { return acquireCanvasTexture(); })
            );

            if (g_verboseLogging) std::cout << "[Canvas] WebGPU context created" << std::endl;
            return canvasContext;
        })
    );

    // Set global canvas - this is the SAME object as document.getElementById('canvas')
    // so it now has both WebGPU getContext AND event listener support
    engine->setGlobalProperty("canvas", canvasObject);

    // ========================================================================
    // Add missing methods to the existing document (from runtime.cpp)
    // We DON'T create a new document - just augment the existing one
    // ========================================================================

    // Add querySelector to existing document (if not present)
    engine->setProperty(existingDocument, "querySelector",
        engine->newFunction("querySelector", [](void* ctx, const std::vector<js::JSValueHandle>& args) {
            // Check if querying for canvas
            if (!args.empty()) {
                std::string selector = g_engine->toString(args[0]);
                if (selector == "canvas" || selector.find("canvas") != std::string::npos) {
                    return g_engine->getGlobalProperty("canvas");
                }
            }
            return g_engine->newNull();
        })
    );

    // Add createElement to existing document
    // NOTE: runtime.cpp sets up a createElement with canvas support (toDataURL) for @loaders.gl WebP detection
    // We ALWAYS override it here to add proper Canvas 2D support for offscreen canvases
    engine->setProperty(existingDocument, "createElement",
        engine->newFunction("createElement", [](void* ctx, const std::vector<js::JSValueHandle>& args) {
            auto element = g_engine->newObject();

            // Get tag name if provided
            std::string tagName = "";
            if (!args.empty()) {
                tagName = g_engine->toString(args[0]);
            }

            // Add basic DOM element properties
            g_engine->setProperty(element, "style", g_engine->newObject());
            g_engine->setProperty(element, "className", g_engine->newString(""));
            g_engine->setProperty(element, "innerHTML", g_engine->newString(""));
            g_engine->setProperty(element, "textContent", g_engine->newString(""));
            g_engine->setProperty(element, "tagName", g_engine->newString(tagName.c_str()));
            g_engine->setProperty(element, "appendChild",
                g_engine->newFunction("appendChild", [](void* c, const std::vector<js::JSValueHandle>& a) {
                    return a.empty() ? g_engine->newUndefined() : a[0];
                })
            );
            g_engine->setProperty(element, "removeChild",
                g_engine->newFunction("removeChild", [](void* c, const std::vector<js::JSValueHandle>& a) {
                    return a.empty() ? g_engine->newUndefined() : a[0];
                })
            );
            g_engine->setProperty(element, "remove",
                g_engine->newFunction("remove", [](void* c, const std::vector<js::JSValueHandle>& a) {
                    // No-op in native runtime - element is not attached to DOM
                    return g_engine->newUndefined();
                })
            );
            g_engine->setProperty(element, "addEventListener",
                g_engine->newFunction("addEventListener", [](void* c, const std::vector<js::JSValueHandle>& a) {
                    // No-op in native runtime
                    return g_engine->newUndefined();
                })
            );
            g_engine->setProperty(element, "removeEventListener",
                g_engine->newFunction("removeEventListener", [](void* c, const std::vector<js::JSValueHandle>& a) {
                    return g_engine->newUndefined();
                })
            );

            // Special handling for canvas elements - add Canvas 2D support
            if (tagName == "canvas" || tagName == "CANVAS") {
                // Create OffscreenCanvas struct to store state
                int canvasId = g_nextOffscreenCanvasId++;
                auto offscreenCanvas = std::make_unique<OffscreenCanvas>();
                OffscreenCanvas* canvasPtr = offscreenCanvas.get();
                g_offscreenCanvases[canvasId] = std::move(offscreenCanvas);

                // Store the canvas ID as private data for getContext lookup
                g_engine->setPrivateData(element, reinterpret_cast<void*>(static_cast<intptr_t>(canvasId)));

                // Also store as property for debugging
                g_engine->setProperty(element, "_offscreenCanvasId", g_engine->newNumber(canvasId));

                // Default canvas dimensions (stored in struct)
                g_engine->setProperty(element, "width", g_engine->newNumber(canvasPtr->width));
                g_engine->setProperty(element, "height", g_engine->newNumber(canvasPtr->height));

                // Store reference to element globally so getContext can find it
                std::string globalName = "__offscreenCanvas_" + std::to_string(canvasId);
                g_engine->setGlobalProperty(globalName.c_str(), element);

                // Create getContext function
                // Capture canvasId to ensure each canvas element's getContext uses its own canvas
                // This fixes the bug where all canvases shared the same context
                auto getContextFn = g_engine->newFunction("getContext", [canvasId, canvasPtr](void* c, const std::vector<js::JSValueHandle>& contextArgs) {
                    if (contextArgs.empty()) {
                        return g_engine->newNull();
                    }

                    std::string contextType = g_engine->toString(contextArgs[0]);

                    // Use the captured canvasId to find the correct canvas
                    // This ensures each canvas element's getContext returns its own context
                    auto it = g_offscreenCanvases.find(canvasId);
                    if (it == g_offscreenCanvases.end()) {
                        std::cerr << "[Canvas] Canvas not found: " << canvasId << std::endl;
                        return g_engine->newNull();
                    }

                    OffscreenCanvas* canvas = it->second.get();

                    if (contextType == "2d") {
                        // Return cached context if already created
                        if (canvas->hasContext2d) {
                            return canvas->context2d;
                        }

                        // Get current dimensions from the canvas element (in case they were changed)
                        std::string globalName = "__offscreenCanvas_" + std::to_string(canvasId);
                        auto canvasElement = g_engine->getGlobalProperty(globalName.c_str());
                        if (!g_engine->isNull(canvasElement) && !g_engine->isUndefined(canvasElement)) {
                            auto widthProp = g_engine->getProperty(canvasElement, "width");
                            auto heightProp = g_engine->getProperty(canvasElement, "height");
                            if (!g_engine->isUndefined(widthProp)) {
                                canvas->width = static_cast<int>(g_engine->toNumber(widthProp));
                            }
                            if (!g_engine->isUndefined(heightProp)) {
                                canvas->height = static_cast<int>(g_engine->toNumber(heightProp));
                            }
                        }

                        // Create Canvas 2D context
                        if (g_verboseLogging) std::cout << "[Canvas] Creating offscreen 2D context (" << canvas->width << "x" << canvas->height << ")" << std::endl;
                        canvas->context2d = canvas::createCanvas2DContext(g_engine, canvas->width, canvas->height);
                        canvas->hasContext2d = true;
                        g_engine->protect(canvas->context2d);
                        return canvas->context2d;
                    }

                    if (contextType == "webgpu") {
                        // Create GPUCanvasContext for offscreen canvas
                        // This shares the main surface/device for simplicity
                        if (g_verboseLogging) std::cout << "[Canvas] Creating offscreen WebGPU context" << std::endl;

                        // Suspend frame tracking - this context persists across frames
                        g_engine->suspendFrameTracking();

                        auto canvasContext = g_engine->newObject();

                        // The shared surface is owned by Context and can retire when the device is destroyed.

                        // context.canvas - reference back to canvas element
                        std::string globalName = "__offscreenCanvas_" + std::to_string(canvasId);
                        auto canvasElement = g_engine->getGlobalProperty(globalName.c_str());
                        g_engine->setProperty(canvasContext, "canvas", canvasElement);

                        // context.configure({ device, format, alphaMode })
                        g_engine->setProperty(canvasContext, "configure",
                            g_engine->newFunction("configure", [](void* ctx, const std::vector<js::JSValueHandle>& args) {
                                if (g_deviceDestroyed) {
                                    g_engine->throwException("The single-device Runtime is destroyed; create a new Runtime");
                                    return g_engine->newUndefined();
                                }
                                if (args.empty()) {
                                    g_engine->throwException("configure requires a descriptor");
                                    return g_engine->newUndefined();
                                }
                                auto descriptor = args[0];
                                std::string format = g_engine->toString(g_engine->getProperty(descriptor, "format"));
                                g_surfaceFormat = stringToFormat(format);
                                g_contextConfigured = true;
                                if (g_verboseLogging) std::cout << "[Canvas] Offscreen context configured with format: " << format << std::endl;
                                return g_engine->newUndefined();
                            })
                        );

                        // context.unconfigure()
                        g_engine->setProperty(canvasContext, "unconfigure",
                            g_engine->newFunction("unconfigure", [](void* ctx, const std::vector<js::JSValueHandle>& args) {
                                return g_engine->newUndefined();
                            })
                        );

                        // context.getCurrentTexture() -> GPUTexture
                        g_engine->setProperty(canvasContext, "getCurrentTexture",
                            g_engine->newFunction("getCurrentTexture", [](void* ctx, const std::vector<js::JSValueHandle>& args) { return acquireCanvasTexture(); })
                        );

                        // Resume frame tracking
                        g_engine->resumeFrameTracking();

                        return canvasContext;
                    }

                    // Ignore webgl requests silently (PixiJS feature detection)
                    if (contextType == "webgl" || contextType == "webgl2" || contextType == "experimental-webgl") {
                        return g_engine->newNull();
                    }

                    std::cerr << "[Canvas] Unsupported context type: " << contextType << std::endl;
                    return g_engine->newNull();
                });

                g_engine->setProperty(element, "getContext", getContextFn);
                if (g_verboseLogging) std::cout << "[Canvas] Created offscreen canvas " << canvasId << std::endl;

                // toDataURL for compatibility (returns empty data URI)
                g_engine->setProperty(element, "toDataURL",
                    g_engine->newFunction("toDataURL", [](void* c, const std::vector<js::JSValueHandle>& a) {
                        std::string mimeType = "image/png";
                        if (!a.empty()) {
                            mimeType = g_engine->toString(a[0]);
                        }
                        // Return a minimal data URI (for @loaders.gl WebP detection)
                        if (mimeType.find("webp") != std::string::npos) {
                            return g_engine->newString("data:image/webp;base64,");
                        }
                        return g_engine->newString("data:image/png;base64,");
                    })
                );

                // getBoundingClientRect - return canvas dimensions
                g_engine->setProperty(element, "getBoundingClientRect",
                    g_engine->newFunction("getBoundingClientRect", [](void* c, const std::vector<js::JSValueHandle>& a) {
                        // Get dimensions from the main canvas if available
                        auto rect = g_engine->newObject();
                        g_engine->setProperty(rect, "x", g_engine->newNumber(0));
                        g_engine->setProperty(rect, "y", g_engine->newNumber(0));
                        g_engine->setProperty(rect, "width", g_engine->newNumber(g_canvasWidth));
                        g_engine->setProperty(rect, "height", g_engine->newNumber(g_canvasHeight));
                        g_engine->setProperty(rect, "top", g_engine->newNumber(0));
                        g_engine->setProperty(rect, "left", g_engine->newNumber(0));
                        g_engine->setProperty(rect, "right", g_engine->newNumber(g_canvasWidth));
                        g_engine->setProperty(rect, "bottom", g_engine->newNumber(g_canvasHeight));
                        return rect;
                    })
                );
            }

            return element;
        })
    );

    // Add document.body if not present, or enhance existing body with required methods
    auto existingBody = engine->getProperty(existingDocument, "body");
    if (engine->isUndefined(existingBody) || engine->isNull(existingBody)) {
        existingBody = engine->newObject();
        engine->setProperty(existingDocument, "body", existingBody);
    }
    // Always add/update these methods on body
    engine->setProperty(existingBody, "style", engine->newObject());
    engine->setProperty(existingBody, "appendChild",
        engine->newFunction("appendChild", [](void* ctx, const std::vector<js::JSValueHandle>& args) {
            return args.empty() ? g_engine->newUndefined() : args[0];
        })
    );
    engine->setProperty(existingBody, "removeChild",
        engine->newFunction("removeChild", [](void* ctx, const std::vector<js::JSValueHandle>& args) {
            return args.empty() ? g_engine->newUndefined() : args[0];
        })
    );

    // ========================================================================
    // Navigator object
    // ========================================================================
    auto navigatorHandle = engine->getGlobalProperty("navigator");
    if (engine->isUndefined(navigatorHandle)) {
        navigatorHandle = engine->newObject();
        engine->setGlobalProperty("navigator", navigatorHandle);
    }

    // Add common navigator properties for browser compatibility
    // PixiJS and other libraries check these for feature detection
    engine->setProperty(navigatorHandle, "userAgent",
        engine->newString("Mozilla/5.0 (Macintosh; MystralNative/0.1) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"));
    engine->setProperty(navigatorHandle, "platform", engine->newString("MystralNative"));
    engine->setProperty(navigatorHandle, "vendor", engine->newString("Mystral Engine"));
    engine->setProperty(navigatorHandle, "language", engine->newString("en-US"));
    engine->setProperty(navigatorHandle, "languages", engine->newArray(1));  // ["en-US"]
    engine->setProperty(navigatorHandle, "onLine", engine->newBoolean(true));
    engine->setProperty(navigatorHandle, "hardwareConcurrency", engine->newNumber(8));
    engine->setProperty(navigatorHandle, "maxTouchPoints", engine->newNumber(0));

    // Create navigator.gpu object
    auto gpuObject = engine->newObject();

    // ========================================================================
    // navigator.gpu.requestAdapter()
    // ========================================================================
    engine->setProperty(gpuObject, "requestAdapter",
        engine->newFunction("requestAdapter", [](void* ctx, const std::vector<js::JSValueHandle>& args) {
            // Expose the adapter selected by the native context.
            auto adapter = g_engine->newObject();

            // adapter.requestDevice()
            g_engine->setProperty(adapter, "requestDevice",
                g_engine->newFunction("requestDevice", [](void* ctx, const std::vector<js::JSValueHandle>& args) {
                    const auto requestError = validateDeviceRequest(args);
                    if (!requestError.empty()) return rejected(requestError);
                    if (g_jsDevice.ptr) return resolved(g_jsDevice);
                    // One native context owns this device; every wrapper observes the same loss/event state.
                    auto device = g_engine->evalScriptWithResult("new EventTarget()", "<GPUDevice event target>");
                    g_engine->setPrivateData(device, g_device);

                    // device.queue
                    auto queue = g_engine->newObject();
                    g_engine->setPrivateData(queue, g_queue);

                    // queue.submit(commandBuffers)
                    g_engine->setProperty(queue, "submit",
                        g_engine->newFunction("submit", [](void* ctx, const std::vector<js::JSValueHandle>& args) {
                            if (args.empty()) {
                                return g_engine->newUndefined();
                            }

                            // Get command buffers array and submit them
                            auto cmdBuffersArray = args[0];

                            // Get array length
                            auto lengthProp = g_engine->getProperty(cmdBuffersArray, "length");
                            int length = (int)g_engine->toNumber(lengthProp);

                            // Collect command buffers
                            std::vector<WGPUCommandBuffer> cmdBuffers;
                            std::unordered_set<WGPUCommandBuffer> uniqueBuffers;
                            for (int i = 0; i < length; i++) {
                                auto cmdBufferHandle = g_engine->getPropertyIndex(cmdBuffersArray, i);
                                WGPUCommandBuffer cmdBuffer = (WGPUCommandBuffer)resourceData(cmdBufferHandle);
                                if (!cmdBuffer || !uniqueBuffers.insert(cmdBuffer).second) {
                                    g_engine->throwException("Command buffer is invalid, already submitted, or duplicated");
                                    return g_engine->newUndefined();
                                }
                                cmdBuffers.push_back(cmdBuffer);
                            }

                            // Submit user command buffers first
                            static int submitCount = 0;
                            submitCount++;
                            if (g_queue) {
                                wgpuQueueSubmit(g_queue, cmdBuffers.size(), cmdBuffers.data());
                                // Release command buffers after submission (they're consumed by submit)
                                for (auto cmdBuf : cmdBuffers) {
                                    releaseReference(cmdBuf, ResourceKind::CommandBuffer);
                                }
                                // Tick to flush GPU work
#if defined(MYSTRAL_WEBGPU_DAWN)
                                wgpuDeviceTick(g_device);
#elif defined(MYSTRAL_WEBGPU_WGPU)
                                wgpuDevicePoll(g_device, false, nullptr);
#endif
                                if (g_verboseLogging) std::cout << "[WebGPU] Submit #" << submitCount << ": " << cmdBuffers.size() << " command buffers, g_currentTexture=" << (void*)g_currentTexture << std::endl;
                            } else {
                                if (g_verboseLogging) std::cout << "[WebGPU] Submit #" << submitCount << ": EMPTY (length=" << length << "), g_currentTexture=" << (void*)g_currentTexture << std::endl;
                            }

                            // Copy texture to screenshot buffer ONLY when about to present
                            // This prevents capturing intermediate render passes (e.g., Three.js post-processing)
                            // Only capture when the surface render pass has ended, matching the present condition
                            // Also ensure we only capture once per frame (Three.js does multiple queue.submit() per frame)
                            WGPUTexture screenshotTexture = g_currentViewSourceTexture ? g_currentViewSourceTexture : g_currentTexture;
                            if (!g_canvasFrameDiscarded && g_surfaceRenderPassEnded && !g_screenshotCapturedThisFrame && screenshotTexture && g_device && g_queue) {
                                // Calculate buffer requirements
                                uint32_t bytesPerPixel = 4;  // BGRA8
                                uint32_t unalignedBytesPerRow = g_canvasWidth * bytesPerPixel;
                                uint32_t bytesPerRow = (unalignedBytesPerRow + 255) & ~255;  // Align to 256
                                size_t requiredSize = bytesPerRow * g_canvasHeight;

                                // Create or resize screenshot buffer if needed
                                if (!g_screenshotBuffer || g_screenshotBufferSize < requiredSize) {
                                    if (g_screenshotBuffer) {
                                        wgpuBufferDestroy(g_screenshotBuffer);
                                        wgpuBufferRelease(g_screenshotBuffer);
                                    }

                                    WGPUBufferDescriptor bufferDesc = {};
                                    bufferDesc.size = requiredSize;
                                    bufferDesc.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead;
                                    bufferDesc.mappedAtCreation = false;

                                    g_screenshotBuffer = wgpuDeviceCreateBuffer(g_device, &bufferDesc);
                                    g_screenshotBufferSize = requiredSize;
                                    g_screenshotBytesPerRow = bytesPerRow;
                                }

                                // Create encoder to copy texture to buffer
                                WGPUCommandEncoderDescriptor encDesc = {};
                                WGPUCommandEncoder copyEncoder = wgpuDeviceCreateCommandEncoder(g_device, &encDesc);

                                WGPUImageCopyTexture_Compat srcCopy = {};
                                srcCopy.texture = screenshotTexture;
                                srcCopy.mipLevel = 0;
                                srcCopy.origin = {0, 0, 0};
                                srcCopy.aspect = WGPUTextureAspect_All;

                                WGPUImageCopyBuffer_Compat dstCopy = {};
                                dstCopy.buffer = g_screenshotBuffer;
                                dstCopy.layout.offset = 0;
                                dstCopy.layout.bytesPerRow = bytesPerRow;
                                dstCopy.layout.rowsPerImage = g_canvasHeight;

                                WGPUExtent3D copySize = {g_canvasWidth, g_canvasHeight, 1};
                                wgpuCommandEncoderCopyTextureToBuffer(copyEncoder, &srcCopy, &dstCopy, &copySize);

                                if (g_verboseLogging) std::cout << "[Screenshot] Copying from texture " << (void*)screenshotTexture
                                          << " (format=" << g_surfaceFormat << ", size=" << g_canvasWidth << "x" << g_canvasHeight << ")" << std::endl;

                                WGPUCommandBufferDescriptor cmdDesc = {};
                                WGPUCommandBuffer copyCmd = wgpuCommandEncoderFinish(copyEncoder, &cmdDesc);
                                wgpuQueueSubmit(g_queue, 1, &copyCmd);

                                wgpuCommandBufferRelease(copyCmd);
                                wgpuCommandEncoderRelease(copyEncoder);

                                // Wait for GPU work to complete before present
                                // This ensures the screenshot copy finishes before the texture is released
                                for (int syncIter = 0; syncIter < 100; syncIter++) {
#if defined(MYSTRAL_WEBGPU_DAWN)
                                    wgpuDeviceTick(g_device);
                                    if (g_instance) {
                                        wgpuInstanceProcessEvents(g_instance);
                                    }
#elif defined(MYSTRAL_WEBGPU_WGPU)
                                    // wgpu-native has no wgpuInstanceProcessEvents (it is an
                                    // unimplemented!() stub that aborts the process);
                                    // wgpuDevicePoll is its equivalent.
                                    wgpuDevicePoll(g_device, false, nullptr);
#endif
                                }

                                g_screenshotReady = true;
                                g_screenshotCapturedThisFrame = true;
                            }

                            // Present the surface only if:
                            // 1. We have a current texture
                            // 2. The surface render pass has been ended (commands submitted)
                            // This prevents presenting before the render commands are in this submit
                            if (!g_canvasFrameDiscarded && g_surface && g_currentTexture && g_surfaceRenderPassEnded) {
                                if (g_verboseLogging) std::cout << "[WebGPU] Presenting surface" << std::endl;
                                wgpuSurfacePresent(g_surface);

                                // Reset surface render tracking for next frame
                                g_surfaceRenderEncoder = nullptr;
                                g_surfaceRenderPassEnded = false;
                                g_screenshotCapturedThisFrame = false;

                                expireCanvasTexture();
                            }

                            return g_engine->newUndefined();
                        })
                    );

                    // queue.writeBuffer(buffer, offset, data, dataOffset?, size?)
                    g_engine->setProperty(queue, "writeBuffer",
                        g_engine->newFunction("writeBuffer", [](void* ctx, const std::vector<js::JSValueHandle>& args) {
                            if (args.size() < 3) {
                                g_engine->throwException("writeBuffer requires buffer, offset, and data");
                                return g_engine->newUndefined();
                            }

                            WGPUBuffer buffer = (WGPUBuffer)resourceData(args[0]);
                            uint64_t offset = (uint64_t)g_engine->toNumber(args[1]);

                            // Get ArrayBuffer data
                            size_t dataSize = 0;
                            void* dataPtr = g_engine->getArrayBufferData(args[2], &dataSize);

                            if (dataSize == 0) {
                                return g_engine->newUndefined();
                            }

                            if (!dataPtr) {
                                g_engine->throwException("writeBuffer: invalid data");
                                return g_engine->newUndefined();
                            }

                            // Optional dataOffset and size
                            size_t dataOffset = args.size() > 3 ? (size_t)g_engine->toNumber(args[3]) : 0;
                            size_t writeSize = args.size() > 4 ? (size_t)g_engine->toNumber(args[4]) : (dataSize - dataOffset);

                            if (buffer && g_queue) {
                                wgpuQueueWriteBuffer(g_queue, buffer, offset, (uint8_t*)dataPtr + dataOffset, writeSize);
                            }

                            return g_engine->newUndefined();
                        })
                    );

                    // queue.writeTexture(destination, data, dataLayout, size)
                    g_engine->setProperty(queue, "writeTexture",
                        g_engine->newFunction("writeTexture", [](void* ctx, const std::vector<js::JSValueHandle>& args) {
                            if (args.size() < 4) {
                                g_engine->throwException("writeTexture requires destination, data, dataLayout, and size");
                                return g_engine->newUndefined();
                            }

                            // Parse destination {texture, mipLevel?, origin?, aspect?}
                            auto destination = args[0];
                            auto textureHandle = g_engine->getProperty(destination, "texture");
                            WGPUTexture texture = (WGPUTexture)resourceData(textureHandle);

                            if (!texture) {
                                g_engine->throwException("writeTexture: invalid texture");
                                return g_engine->newUndefined();
                            }

                            auto mipLevelVal = g_engine->getProperty(destination, "mipLevel");
                            uint32_t mipLevel = g_engine->isUndefined(mipLevelVal) ? 0 : (uint32_t)g_engine->toNumber(mipLevelVal);

                            // Parse origin
                            auto originVal = g_engine->getProperty(destination, "origin");
                            uint32_t originX = 0, originY = 0, originZ = 0;
                            if (!g_engine->isUndefined(originVal)) {
                                auto lengthProp = g_engine->getProperty(originVal, "length");
                                if (!g_engine->isUndefined(lengthProp)) {
                                    // Array format
                                    int len = (int)g_engine->toNumber(lengthProp);
                                    if (len >= 1) originX = (uint32_t)g_engine->toNumber(g_engine->getPropertyIndex(originVal, 0));
                                    if (len >= 2) originY = (uint32_t)g_engine->toNumber(g_engine->getPropertyIndex(originVal, 1));
                                    if (len >= 3) originZ = (uint32_t)g_engine->toNumber(g_engine->getPropertyIndex(originVal, 2));
                                } else {
                                    // Object format
                                    auto x = g_engine->getProperty(originVal, "x");
                                    auto y = g_engine->getProperty(originVal, "y");
                                    auto z = g_engine->getProperty(originVal, "z");
                                    if (!g_engine->isUndefined(x)) originX = (uint32_t)g_engine->toNumber(x);
                                    if (!g_engine->isUndefined(y)) originY = (uint32_t)g_engine->toNumber(y);
                                    if (!g_engine->isUndefined(z)) originZ = (uint32_t)g_engine->toNumber(z);
                                }
                            }

                            // Get ArrayBuffer data
                            size_t dataSize = 0;
                            void* dataPtr = g_engine->getArrayBufferData(args[1], &dataSize);

                            if (!dataPtr || dataSize == 0) {
                                g_engine->throwException("writeTexture: invalid data");
                                return g_engine->newUndefined();
                            }

                            // Parse size FIRST (need height for rowsPerImage default)
                            auto sizeVal = args[3];
                            uint32_t width = 1, height = 1, depthOrArrayLayers = 1;
                            auto lengthProp = g_engine->getProperty(sizeVal, "length");
                            if (!g_engine->isUndefined(lengthProp)) {
                                int len = (int)g_engine->toNumber(lengthProp);
                                if (len >= 1) width = (uint32_t)g_engine->toNumber(g_engine->getPropertyIndex(sizeVal, 0));
                                if (len >= 2) height = (uint32_t)g_engine->toNumber(g_engine->getPropertyIndex(sizeVal, 1));
                                if (len >= 3) depthOrArrayLayers = (uint32_t)g_engine->toNumber(g_engine->getPropertyIndex(sizeVal, 2));
                            } else {
                                auto w = g_engine->getProperty(sizeVal, "width");
                                auto h = g_engine->getProperty(sizeVal, "height");
                                auto d = g_engine->getProperty(sizeVal, "depthOrArrayLayers");
                                if (!g_engine->isUndefined(w)) width = (uint32_t)g_engine->toNumber(w);
                                if (!g_engine->isUndefined(h)) height = (uint32_t)g_engine->toNumber(h);
                                if (!g_engine->isUndefined(d)) depthOrArrayLayers = (uint32_t)g_engine->toNumber(d);
                            }

                            WGPUTextureDataLayout_Compat layout = {};
                            if (!copyLayout(args[2], layout)) return g_engine->newUndefined();

                            // Create copy structures
                            WGPUImageCopyTexture_Compat destCopy = {};
                            destCopy.texture = texture;
                            destCopy.mipLevel = mipLevel;
                            destCopy.origin = {originX, originY, originZ};
                            destCopy.aspect = WGPUTextureAspect_All;

                            WGPUExtent3D copySize = {width, height, depthOrArrayLayers};

                            // Write texture
                            wgpuQueueWriteTexture(g_queue, &destCopy, dataPtr, dataSize, &layout, &copySize);

                            if (g_verboseLogging) std::cout << "[WebGPU] writeTexture: " << width << "x" << height << " (" << dataSize << " bytes)" << std::endl;

                            return g_engine->newUndefined();
                        })
                    );

                    // queue.copyExternalImageToTexture(source, destination, copySize)
                    // Standard WebGPU way to upload ImageBitmap to texture
                    g_engine->setProperty(queue, "copyExternalImageToTexture",
                        g_engine->newFunction("copyExternalImageToTexture", [](void* ctx, const std::vector<js::JSValueHandle>& args) {
                            if (args.size() < 3) {
                                g_engine->throwException("copyExternalImageToTexture requires source, destination, and copySize");
                                return g_engine->newUndefined();
                            }

                            // Parse source (ImageBitmap-like object or canvas element)
                            auto source = args[0];
                            auto sourceObj = g_engine->getProperty(source, "source");
                            if (g_engine->isUndefined(sourceObj)) {
                                sourceObj = source; // source might be passed directly
                            }

                            // Parse flipY option (default false per WebGPU spec)
                            bool flipY = false;
                            auto flipYProp = g_engine->getProperty(source, "flipY");
                            if (!g_engine->isUndefined(flipYProp)) {
                                flipY = g_engine->toBoolean(flipYProp);
                            }

                            // Parse destination.premultipliedAlpha (default false per WebGPU spec).
                            // PixiJS sets this true so its NORMAL blend (ONE, ONE_MINUS_SRC_ALPHA)
                            // produces correct results. Our PNG decoder returns straight alpha, so
                            // when the destination requests premultiplied we must multiply RGB by
                            // A/255 on the fly. Without this, pixels with 0<a<255 and color>0 render
                            // too bright / with color halos.
                            bool premultipliedAlpha = false;
                            {
                                auto premulProp = g_engine->getProperty(args[1], "premultipliedAlpha");
                                if (!g_engine->isUndefined(premulProp)) {
                                    premultipliedAlpha = g_engine->toBoolean(premulProp);
                                }
                            }

                            int imgWidth = 0;
                            int imgHeight = 0;
                            size_t dataSize = 0;
                            void* dataPtr = nullptr;

                            // Try to get data from ImageBitmap
                            auto imageData = g_engine->getProperty(sourceObj, "_data");
                            if (!g_engine->isUndefined(imageData)) {
                                // Standard ImageBitmap with _data
                                imgWidth = (int)g_engine->toNumber(g_engine->getProperty(sourceObj, "width"));
                                imgHeight = (int)g_engine->toNumber(g_engine->getProperty(sourceObj, "height"));
                                dataPtr = g_engine->getArrayBufferData(imageData, &dataSize);
                            } else {
                                // Check if it's a canvas element
                                auto tagName = g_engine->getProperty(sourceObj, "tagName");
                                std::string tagNameStr = g_engine->isUndefined(tagName) ? "" : g_engine->toString(tagName);

                                if (tagNameStr == "CANVAS" || tagNameStr == "canvas") {
                                    // Get the canvas ID from private data or property
                                    auto canvasIdProp = g_engine->getProperty(sourceObj, "_offscreenCanvasId");
                                    if (!g_engine->isUndefined(canvasIdProp)) {
                                        int canvasId = (int)g_engine->toNumber(canvasIdProp);
                                        auto it = g_offscreenCanvases.find(canvasId);
                                        if (it != g_offscreenCanvases.end() && it->second->hasContext2d) {
                                            // Get pixel data from the 2D context
                                            auto ctx2dHandle = it->second->context2d;
                                            auto nativeCtx = static_cast<canvas::Canvas2DContext*>(resourceData(ctx2dHandle));
                                            if (nativeCtx) {
                                                imgWidth = nativeCtx->getWidth();
                                                imgHeight = nativeCtx->getHeight();
                                                dataPtr = const_cast<void*>(static_cast<const void*>(nativeCtx->getPixelData()));
                                                dataSize = nativeCtx->getPixelDataSize();
                                            }
                                        }
                                    }
                                }

                                // Check if it's already a 2D context (has getImageData method or _contextType)
                                auto contextType = g_engine->getProperty(sourceObj, "_contextType");
                                if (!g_engine->isUndefined(contextType)) {
                                    std::string ctxTypeStr = g_engine->toString(contextType);
                                    if (ctxTypeStr == "2d") {
                                        // It's a 2D context, get the canvas and then get pixel data
                                        auto canvas = g_engine->getProperty(sourceObj, "canvas");
                                        if (!g_engine->isUndefined(canvas)) {
                                            auto canvasIdProp = g_engine->getProperty(canvas, "_offscreenCanvasId");
                                            if (!g_engine->isUndefined(canvasIdProp)) {
                                                int canvasId = (int)g_engine->toNumber(canvasIdProp);
                                                auto it = g_offscreenCanvases.find(canvasId);
                                                if (it != g_offscreenCanvases.end() && it->second->hasContext2d) {
                                                    auto nativeCtx = static_cast<canvas::Canvas2DContext*>(resourceData(sourceObj));
                                                    if (nativeCtx) {
                                                        imgWidth = nativeCtx->getWidth();
                                                        imgHeight = nativeCtx->getHeight();
                                                        dataPtr = const_cast<void*>(static_cast<const void*>(nativeCtx->getPixelData()));
                                                        dataSize = nativeCtx->getPixelDataSize();
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            if (!dataPtr || dataSize == 0) {
                                // Try to get width/height anyway for better error message
                                auto widthProp = g_engine->getProperty(sourceObj, "width");
                                auto heightProp = g_engine->getProperty(sourceObj, "height");
                                if (!g_engine->isUndefined(widthProp)) imgWidth = (int)g_engine->toNumber(widthProp);
                                if (!g_engine->isUndefined(heightProp)) imgHeight = (int)g_engine->toNumber(heightProp);

                                std::cerr << "[WebGPU] copyExternalImageToTexture: unsupported source type, width=" << imgWidth << ", height=" << imgHeight << std::endl;
                                // Return silently instead of throwing - PixiJS might be able to continue
                                return g_engine->newUndefined();
                            }

                            if (imgWidth <= 0 || imgHeight <= 0 ||
                                static_cast<uint64_t>(imgWidth) * static_cast<uint64_t>(imgHeight) > dataSize / 4) {
                                g_engine->throwException("External image dimensions exceed the available RGBA pixel data");
                                return g_engine->newUndefined();
                            }

                            // Parse destination
                            auto destination = args[1];
                            auto textureObj = g_engine->getProperty(destination, "texture");
                            WGPUTexture texture = (WGPUTexture)resourceData(textureObj);
                            if (!texture) {
                                g_engine->throwException("copyExternalImageToTexture: invalid texture");
                                return g_engine->newUndefined();
                            }

                            // Detect the destination texture format. In a real browser,
                            // copyExternalImageToTexture converts the source's RGBA pixels into the
                            // destination's format; we upload bytes verbatim via writeTexture, so for
                            // BGRA8 destinations we must swap the R/B channels ourselves. Our
                            // ImageBitmap data is always RGBA (stb_image / WebPDecodeRGBA), but PixiJS
                            // v8's TextureSource.defaultOptions.format is "bgra8unorm", so every
                            // Texture.from(imageBitmap) lands here — without the swap, red and blue
                            // come out transposed.
                            bool swapRB = false;
                            {
                                auto fmtProp = g_engine->getProperty(textureObj, "format");
                                if (!g_engine->isUndefined(fmtProp)) {
                                    std::string fmt = g_engine->toString(fmtProp);
                                    swapRB = (fmt == "bgra8unorm" || fmt == "bgra8unorm-srgb");
                                }
                            }

                            WGPUImageCopyTexture_Compat destCopy = {};
                            if (!copyTexture(destination, destCopy)) return g_engine->newUndefined();

                            // Parse copySize
                            auto sizeVal = args[2];
                            uint32_t width = imgWidth, height = imgHeight, depthOrArrayLayers = 1;
                            if (g_engine->isArray(sizeVal)) {
                                width = (uint32_t)g_engine->toNumber(g_engine->getPropertyIndex(sizeVal, 0));
                                height = (uint32_t)g_engine->toNumber(g_engine->getPropertyIndex(sizeVal, 1));
                                auto depthVal = g_engine->getPropertyIndex(sizeVal, 2);
                                if (!g_engine->isUndefined(depthVal)) {
                                    depthOrArrayLayers = (uint32_t)g_engine->toNumber(depthVal);
                                }
                            } else if (!g_engine->isUndefined(sizeVal)) {
                                auto widthVal = g_engine->getProperty(sizeVal, "width");
                                auto heightVal = g_engine->getProperty(sizeVal, "height");
                                if (!g_engine->isUndefined(widthVal)) width = (uint32_t)g_engine->toNumber(widthVal);
                                if (!g_engine->isUndefined(heightVal)) height = (uint32_t)g_engine->toNumber(heightVal);
                            }

                            if (width > static_cast<uint32_t>(imgWidth) || height > static_cast<uint32_t>(imgHeight) || depthOrArrayLayers > 1) {
                                g_engine->throwException("External image copy exceeds the source pixel surface");
                                return g_engine->newUndefined();
                            }

                            // Handle flipY, premultipliedAlpha, and/or BGRA channel swap by writing
                            // into a staging copy. RGBA8 only (matches the hardcoded bytesPerRow below).
                            std::vector<uint8_t> stagingData;
                            void* uploadDataPtr = dataPtr;
                            if ((flipY || premultipliedAlpha || swapRB) && dataPtr && imgHeight > 0 && imgWidth > 0) {
                                size_t bytesPerRow = (size_t)imgWidth * 4;
                                stagingData.resize(dataSize);
                                const uint8_t* srcData = static_cast<const uint8_t*>(dataPtr);
                                for (int y = 0; y < imgHeight; y++) {
                                    const uint8_t* srcRow = srcData + (flipY ? (imgHeight - 1 - y) : y) * bytesPerRow;
                                    uint8_t* dstRow = stagingData.data() + (size_t)y * bytesPerRow;
                                    if (premultipliedAlpha || swapRB) {
                                        for (int x = 0; x < imgWidth; x++) {
                                            uint32_t r = srcRow[x * 4 + 0];
                                            uint32_t g = srcRow[x * 4 + 1];
                                            uint32_t b = srcRow[x * 4 + 2];
                                            uint32_t a = srcRow[x * 4 + 3];
                                            if (premultipliedAlpha) {
                                                // (v * a + 127) / 255 rounds correctly without a divide instruction
                                                r = (r * a + 127) / 255;
                                                g = (g * a + 127) / 255;
                                                b = (b * a + 127) / 255;
                                            }
                                            // BGRA8 destinations read byte 0 as B and byte 2 as R, so emit
                                            // the channels swapped; RGBA8 destinations get them in order.
                                            dstRow[x * 4 + 0] = (uint8_t)(swapRB ? b : r);
                                            dstRow[x * 4 + 1] = (uint8_t)g;
                                            dstRow[x * 4 + 2] = (uint8_t)(swapRB ? r : b);
                                            dstRow[x * 4 + 3] = (uint8_t)a;
                                        }
                                    } else {
                                        std::memcpy(dstRow, srcRow, bytesPerRow);
                                    }
                                }
                                uploadDataPtr = stagingData.data();
                                if (g_verboseLogging) {
                                    std::cout << "[WebGPU] copyExternalImageToTexture: "
                                              << (flipY ? "flipY " : "")
                                              << (premultipliedAlpha ? "premultiplyAlpha " : "")
                                              << (swapRB ? "swapRB" : "")
                                              << std::endl;
                                }
                            }

                            // Use writeTexture internally (same effect as copyExternalImageToTexture)
                            WGPUTextureDataLayout_Compat layout = {};
                            layout.offset = 0;
                            layout.bytesPerRow = imgWidth * 4;  // RGBA
                            layout.rowsPerImage = imgHeight;

                            WGPUExtent3D copySize = {width, height, depthOrArrayLayers};

                            wgpuQueueWriteTexture(g_queue, &destCopy, uploadDataPtr, dataSize, &layout, &copySize);

                            if (g_verboseLogging) std::cout << "[WebGPU] copyExternalImageToTexture: " << width << "x" << height << (flipY ? " (flipY)" : "") << std::endl;

                            return g_engine->newUndefined();
                        })
                    );

                    // queue.onSubmittedWorkDone() - returns Promise that resolves when GPU work is done
                    g_engine->setProperty(queue, "onSubmittedWorkDone",
                        g_engine->newFunction("onSubmittedWorkDone", [](void* ctx, const std::vector<js::JSValueHandle>& args) {
                            auto deferred = makeDeferred();
#if defined(MYSTRAL_WEBGPU_DAWN)
                            WGPUQueueWorkDoneCallbackInfo callback = {};
                            callback.mode = WGPUCallbackMode_AllowProcessEvents;
                            callback.callback = onQueueCompleted;
                            callback.userdata1 = new std::shared_ptr<DeferredResult>(deferred);
                            wgpuQueueOnSubmittedWorkDone(g_queue, callback);
#else
                            settle(deferred, false, nativeError(WGPUErrorType_Internal, "Native queue completion binding is unavailable for this backend"));
#endif
                            return deferred->promise;
                        })
                    );

                    g_engine->setProperty(device, "queue", queue);

                    // device.limits - expose device limits
                    auto deviceLimits = nativeLimits(false);
                    g_engine->setProperty(device, "limits", deviceLimits);

                    // device.features - Set-like object with enabled features
                    // These should match the features exposed in adapter.features that were requested
                    auto deviceFeatures = nativeFeatures(false);
                    g_engine->setProperty(device, "features", deviceFeatures);

                    // device.createBuffer(descriptor)
                    g_engine->setProperty(device, "createBuffer",
                        g_engine->newFunction("createBuffer", [](void* ctx, const std::vector<js::JSValueHandle>& args) {
                            if (args.empty()) {
                                g_engine->throwException("createBuffer requires a descriptor");
                                return g_engine->newUndefined();
                            }

                            auto descriptor = args[0];
                            double size = g_engine->toNumber(g_engine->getProperty(descriptor, "size"));
                            double usage = g_engine->toNumber(g_engine->getProperty(descriptor, "usage"));

                            // Check for mappedAtCreation
                            auto mappedAtCreationProp = g_engine->getProperty(descriptor, "mappedAtCreation");
                            bool mappedAtCreation = !g_engine->isUndefined(mappedAtCreationProp) && g_engine->toBoolean(mappedAtCreationProp);

                            WGPUBufferDescriptor bufferDesc = {};
                            bufferDesc.size = (uint64_t)size;
                            bufferDesc.usage = (WGPUBufferUsage)(uint32_t)usage;
                            bufferDesc.mappedAtCreation = mappedAtCreation;

                            WGPUBuffer buffer = wgpuDeviceCreateBuffer(g_device, &bufferDesc);
                            if (!buffer) {
                                g_engine->throwException("Failed to create buffer");
                                return g_engine->newUndefined();
                            }

                            // Register buffer for mapping operations
                            uint64_t bufferId = g_nextBufferId++;
                            // mappedAtCreation buffers are mapped for write
                            WGPUMapMode initialMapMode = mappedAtCreation ? WGPUMapMode_Write : WGPUMapMode_None;
                            g_bufferRegistry[bufferId] = {buffer, (uint64_t)size, (WGPUBufferUsage)(uint32_t)usage, mappedAtCreation, nullptr, 0, initialMapMode};

                            auto jsBuffer = g_engine->newObject();
                            manageResource(jsBuffer, buffer, ResourceKind::Buffer, bufferId);
                            g_engine->setProperty(jsBuffer, "size", g_engine->newNumber(size));
                            g_engine->setProperty(jsBuffer, "_bufferId", g_engine->newNumber((double)bufferId));
                            g_engine->setProperty(jsBuffer, "usage", g_engine->newNumber(usage));

                            // Set initial mapState
                            g_engine->setProperty(jsBuffer, "mapState", g_engine->newString(mappedAtCreation ? "mapped" : "unmapped"));

                            // buffer.mapAsync(mode, offset?, size?) -> Promise
                            // Returns a Promise that resolves when the buffer is mapped
                            g_engine->setProperty(jsBuffer, "mapAsync",
                                g_engine->newFunction("mapAsync", [bufferId](void* ctx, const std::vector<js::JSValueHandle>& args) {
                                    auto it = g_bufferRegistry.find(bufferId);
                                    if (it == g_bufferRegistry.end()) {
                                        std::cerr << "[WebGPU] mapAsync: Buffer " << bufferId << " not found" << std::endl;
                                        return g_engine->evalScriptWithResult("Promise.reject(new Error('Buffer not found'))", "mapAsync-error");
                                    }

                                    auto& bufferInfo = it->second;

                                    // Already mapped (mappedAtCreation)?
                                    if (bufferInfo.isMapped) {
                                        return g_engine->evalScriptWithResult("Promise.resolve()", "mapAsync-already-mapped");
                                    }

                                    // Get mode (default to READ)
                                    WGPUMapMode mode = WGPUMapMode_Read;
                                    if (!args.empty()) {
                                        uint32_t jsMode = (uint32_t)g_engine->toNumber(args[0]);
                                        // GPUMapMode.READ = 1, GPUMapMode.WRITE = 2
                                        if (jsMode == 2) mode = WGPUMapMode_Write;
                                    }

                                    uint64_t offset = args.size() > 1 ? (uint64_t)g_engine->toNumber(args[1]) : 0;
                                    uint64_t mapSize = args.size() > 2 ? (uint64_t)g_engine->toNumber(args[2]) : bufferInfo.size;

                                    // Debug: Log buffer info
                                    bool hasMapRead = (bufferInfo.usage & WGPUBufferUsage_MapRead) != 0;
                                    (void)hasMapRead;  // Used for debug logging when enabled

                                    // Ensure all pending GPU work is processed before attempting to map
                                    // This is critical for buffers that were just used in a copy operation
                                    for (int prePoll = 0; prePoll < 100; prePoll++) {
#if defined(MYSTRAL_WEBGPU_WGPU)
                                        wgpuDevicePoll(g_device, false, nullptr);
#else
                                        if (g_instance) {
                                            wgpuInstanceProcessEvents(g_instance);
                                        }
                                        if (g_device) {
                                            wgpuDeviceTick(g_device);
                                        }
#endif
                                    }

                                    // Synchronous mapping: use global callback + device poll
                                    g_bufferMapData.completed = false;
                                    g_bufferMapData.status = WGPUBufferMapAsyncStatus_Unknown_Compat;
                                    g_bufferMapData.errorMessage.clear();

#if WGPU_BUFFER_MAP_USES_CALLBACK_INFO
                                    // Dawn uses CallbackInfo struct with 4-param callback
                                    // The mapping loop drives ProcessEvents on the JS thread that owns this state.
                                    WGPUBufferMapCallbackInfo mapCallbackInfo = {};
                                    mapCallbackInfo.mode = WGPUCallbackMode_AllowProcessEvents;
                                    mapCallbackInfo.callback = onBufferMapped;
                                    mapCallbackInfo.userdata1 = &g_bufferMapData;
                                    mapCallbackInfo.userdata2 = nullptr;

                                    wgpuBufferMapAsync(bufferInfo.buffer, mode, offset, mapSize, mapCallbackInfo);
#else
                                    // wgpu-native uses separate callback and userdata
                                    wgpuBufferMapAsync(bufferInfo.buffer, mode, offset, mapSize, onBufferMapped, &g_bufferMapData);
#endif

                                    // Poll device until mapping completes
                                    // Add small sleep to avoid busy-looping and let GPU work complete
                                    int pollCount = 0;
                                    while (!g_bufferMapData.completed && pollCount < 10000) {
#if defined(MYSTRAL_WEBGPU_WGPU)
                                        wgpuDevicePoll(g_device, true, nullptr);
#else
                                        if (g_instance) {
                                            wgpuInstanceProcessEvents(g_instance);
                                        }
                                        if (g_device) {
                                            wgpuDeviceTick(g_device);
                                        }
#endif
                                        // Small sleep every 100 iterations to avoid busy loop
                                        if (pollCount % 100 == 0) {
                                            std::this_thread::sleep_for(std::chrono::milliseconds(1));
                                        }
                                        pollCount++;
                                    }

                                    if (g_bufferMapData.status == WGPUBufferMapAsyncStatus_Success_Compat) {
                                        bufferInfo.isMapped = true;
                                        bufferInfo.mapMode = mode;  // Store whether mapped for read or write
                                        return g_engine->evalScriptWithResult("Promise.resolve()", "mapAsync-success");
                                    } else {
                                        std::cerr << "[WebGPU] mapAsync: Failed with status " << g_bufferMapData.status;
                                        if (!g_bufferMapData.errorMessage.empty()) {
                                            std::cerr << " - " << g_bufferMapData.errorMessage;
                                        }
                                        std::cerr << std::endl;
                                        return g_engine->evalScriptWithResult("Promise.reject(new Error('Buffer map failed'))", "mapAsync-failed");
                                    }
                                })
                            );

                            // buffer.getMappedRange(offset?, size?) -> ArrayBuffer
                            // Capture bufferId in closure to identify the correct buffer
                            g_engine->setProperty(jsBuffer, "getMappedRange",
                                g_engine->newFunction("getMappedRange", [bufferId](void* ctx, const std::vector<js::JSValueHandle>& args) {
                                    // Look up this specific buffer by its ID
                                    auto it = g_bufferRegistry.find(bufferId);
                                    if (it == g_bufferRegistry.end()) {
                                        std::cerr << "[WebGPU] getMappedRange: Buffer " << bufferId << " not found in registry" << std::endl;
                                        return g_engine->newUndefined();
                                    }

                                    auto& bufferInfo = it->second;

                                    if (!bufferInfo.isMapped && !bufferInfo.mappedData) {
                                        if (g_verboseLogging) std::cerr << "[WebGPU] getMappedRange: Buffer " << bufferId << " is not mapped" << std::endl;
                                        return g_engine->newUndefined();
                                    }

                                    uint64_t offset = args.empty() ? 0 : (uint64_t)g_engine->toNumber(args[0]);
                                    uint64_t rangeSize = args.size() > 1 ? (uint64_t)g_engine->toNumber(args[1]) : bufferInfo.size - offset;

                                    // Use wgpuBufferGetConstMappedRange for MAP_READ, wgpuBufferGetMappedRange for MAP_WRITE
                                    // Dawn requires the const version for read-only mapped buffers
                                    const void* mappedData = nullptr;
                                    if (bufferInfo.mapMode == WGPUMapMode_Read) {
                                        mappedData = wgpuBufferGetConstMappedRange(bufferInfo.buffer, offset, rangeSize);
                                    } else {
                                        mappedData = wgpuBufferGetMappedRange(bufferInfo.buffer, offset, rangeSize);
                                    }

                                    if (mappedData) {
                                        // Use newArrayBufferExternal to avoid copying
                                        // Cast away const for read-only buffers - the JS side shouldn't modify but we need void*
                                        return g_engine->newArrayBufferExternal(const_cast<void*>(mappedData), rangeSize);
                                    }

                                    if (g_verboseLogging) std::cerr << "[WebGPU] getMappedRange: GetMappedRange returned null for buffer " << bufferId << std::endl;
                                    return g_engine->newUndefined();
                                })
                            );

                            // buffer.unmap()
                            // Capture bufferId in closure to identify the correct buffer
                            g_engine->setProperty(jsBuffer, "unmap",
                                g_engine->newFunction("unmap", [bufferId](void* ctx, const std::vector<js::JSValueHandle>& args) {
                                    // Look up this specific buffer by its ID
                                    auto it = g_bufferRegistry.find(bufferId);
                                    if (it == g_bufferRegistry.end()) {
                                        std::cerr << "[WebGPU] unmap: Buffer " << bufferId << " not found in registry" << std::endl;
                                        return g_engine->newUndefined();
                                    }

                                    auto& bufferInfo = it->second;
                                    if (bufferInfo.isMapped) {
                                        wgpuBufferUnmap(bufferInfo.buffer);
                                        bufferInfo.isMapped = false;
                                        bufferInfo.mappedData = nullptr;
                                        bufferInfo.mappedSize = 0;
                                    }
                                    return g_engine->newUndefined();
                                })
                            );

                            // buffer.destroy()
                            // Capture bufferId in closure to identify the correct buffer
                            g_engine->setProperty(jsBuffer, "destroy",
                                g_engine->newFunction("destroy", [bufferId](void* ctx, const std::vector<js::JSValueHandle>& args) {
                                    auto it = g_bufferRegistry.find(bufferId);
                                    if (it != g_bufferRegistry.end()) {
                                        releaseResource(it->second.resourceId, true);
                                    }
                                    return g_engine->newUndefined();
                                })
                            );

                            return jsBuffer;
                        })
                    );

                    // device.createShaderModule(descriptor)
                    g_engine->setProperty(device, "createShaderModule",
                        g_engine->newFunction("createShaderModule", [](void* ctx, const std::vector<js::JSValueHandle>& args) {
                            if (args.empty()) {
                                g_engine->throwException("createShaderModule requires a descriptor");
                                return g_engine->newUndefined();
                            }

                            auto descriptor = args[0];
                            std::string code = g_engine->toString(g_engine->getProperty(descriptor, "code"));

                            // Debug: Print first 500 chars of shader code
                            if (g_verboseLogging && code.length() > 0) {
                                std::cout << "[Shader] Creating shader (" << code.length() << " chars):\n"
                                          << code.substr(0, std::min((size_t)500, code.length()))
                                          << (code.length() > 500 ? "\n..." : "") << std::endl;
                            }

                            WGPUShaderModuleWGSLDescriptor_Compat wgslDesc = {};
                            WGPUShaderModuleDescriptor shaderDesc = {};
                            setupShaderModuleWGSL(&shaderDesc, &wgslDesc, code.c_str());

                            WGPUShaderModule shaderModule = wgpuDeviceCreateShaderModule(g_device, &shaderDesc);

                            auto jsShader = g_engine->newObject();
                            const auto shaderResourceId = manageResource(jsShader, shaderModule, ResourceKind::ShaderModule);
                            g_engine->setProperty(jsShader, "getCompilationInfo",
                                g_engine->newFunction("getCompilationInfo", [shaderResourceId](void*, const std::vector<js::JSValueHandle>&) {
                                    auto shaderModule = resourceHandle<WGPUShaderModule>(shaderResourceId);
                                    if (!shaderModule) return g_engine->newUndefined();
                                    auto deferred = makeDeferred();
#if defined(MYSTRAL_WEBGPU_DAWN)
                                    WGPUCompilationInfoCallbackInfo callback = {};
                                    callback.mode = WGPUCallbackMode_AllowProcessEvents;
                                    callback.callback = onCompilationCompleted;
                                    callback.userdata1 = new std::shared_ptr<DeferredResult>(deferred);
                                    wgpuShaderModuleGetCompilationInfo(shaderModule, callback);
#else
                                    settle(deferred, false, nativeError(WGPUErrorType_Internal, "Native compilation information binding is unavailable for this backend"));
#endif
                                    return deferred->promise;
                                }));

                            return jsShader;
                        })
                    );

                    // device.createRenderPipeline(descriptor)
                    g_engine->setProperty(device, "createRenderPipeline",
                        g_engine->newFunction("createRenderPipeline", [](void* ctx, const std::vector<js::JSValueHandle>& args) {
                            if (args.empty()) {
                                g_engine->throwException("createRenderPipeline requires a descriptor");
                                return g_engine->newUndefined();
                            }

                            auto descriptor = args[0];

                            // Get vertex stage
                            auto vertex = g_engine->getProperty(descriptor, "vertex");
                            auto vertexModule = g_engine->getProperty(vertex, "module");
                            std::string vertexEntry = g_engine->toString(g_engine->getProperty(vertex, "entryPoint"));

                            // Get fragment stage (optional - depth-only pipelines don't have fragment)
                            auto fragment = g_engine->getProperty(descriptor, "fragment");
                            WGPUShaderModule fsModule = nullptr;
                            std::string fragmentEntry = "main";
                            bool hasFragment = !g_engine->isUndefined(fragment) && !g_engine->isNull(fragment);
                            if (hasFragment) {
                                auto fragmentModule = g_engine->getProperty(fragment, "module");
                                fsModule = (WGPUShaderModule)resourceData(fragmentModule);
                                auto fragEntryProp = g_engine->getProperty(fragment, "entryPoint");
                                if (!g_engine->isUndefined(fragEntryProp)) {
                                    fragmentEntry = g_engine->toString(fragEntryProp);
                                }
                            }

                            // Get native shader modules
                            WGPUShaderModule vsModule = (WGPUShaderModule)resourceData(vertexModule);

                            // Create pipeline descriptor
                            WGPURenderPipelineDescriptor pipelineDesc = {};

                            // Check for layout property
                            auto layoutProp = g_engine->getProperty(descriptor, "layout");
                            if (!g_engine->isUndefined(layoutProp)) {
                                // Check if it's "auto" string or a PipelineLayout object
                                if (g_engine->isString(layoutProp)) {
                                    std::string layoutStr = g_engine->toString(layoutProp);
                                    if (layoutStr == "auto") {
                                        pipelineDesc.layout = nullptr;  // Auto layout
                                    }
                                } else {
                                    // It's a PipelineLayout object
                                    WGPUPipelineLayout layout = (WGPUPipelineLayout)resourceData(layoutProp);
                                    pipelineDesc.layout = layout;
                                }
                            }

                            // Vertex state
                            pipelineDesc.vertex.module = vsModule;
                            WGPU_SET_ENTRY_POINT(pipelineDesc.vertex, vertexEntry.c_str());

                            // Parse vertex buffers if present
                            std::vector<WGPUVertexBufferLayout> vertexBuffers;
                            std::vector<std::vector<WGPUVertexAttribute>> allAttributes; // Keep attributes alive

                            auto buffersArray = g_engine->getProperty(vertex, "buffers");
                            if (!g_engine->isUndefined(buffersArray)) {
                                if (!sequenceArray(buffersArray, buffersArray)) return g_engine->newUndefined();
                                auto buffersLen = g_engine->getProperty(buffersArray, "length");
                                int bufferCount = (int)g_engine->toNumber(buffersLen);

                                for (int i = 0; i < bufferCount; i++) {
                                    auto buffer = g_engine->getPropertyIndex(buffersArray, i);

                                    WGPUVertexBufferLayout layout = {};
                                    layout.arrayStride = (uint64_t)g_engine->toNumber(g_engine->getProperty(buffer, "arrayStride"));
                                    layout.stepMode = WGPUVertexStepMode_Vertex;

                                    // Parse step mode if present
                                    auto stepModeProp = g_engine->getProperty(buffer, "stepMode");
                                    if (!g_engine->isUndefined(stepModeProp)) {
                                        std::string stepModeStr = g_engine->toString(stepModeProp);
                                        if (stepModeStr == "instance") {
                                            layout.stepMode = WGPUVertexStepMode_Instance;
                                        }
                                    }

                                    // Parse attributes
                                    auto attrsArray = g_engine->getProperty(buffer, "attributes");
                                    if (!g_engine->isUndefined(attrsArray)) {
                                        if (!sequenceArray(attrsArray, attrsArray)) return g_engine->newUndefined();
                                        auto attrsLen = g_engine->getProperty(attrsArray, "length");
                                        int attrCount = (int)g_engine->toNumber(attrsLen);

                                        std::vector<WGPUVertexAttribute> attributes;
                                        for (int j = 0; j < attrCount; j++) {
                                            auto attr = g_engine->getPropertyIndex(attrsArray, j);

                                            WGPUVertexAttribute va = {};
                                            va.shaderLocation = (uint32_t)g_engine->toNumber(g_engine->getProperty(attr, "shaderLocation"));
                                            va.offset = (uint64_t)g_engine->toNumber(g_engine->getProperty(attr, "offset"));

                                            std::string formatStr = g_engine->toString(g_engine->getProperty(attr, "format"));
                                            // Parse vertex format
                                            if (formatStr == "float32") va.format = WGPUVertexFormat_Float32;
                                            else if (formatStr == "float32x2") va.format = WGPUVertexFormat_Float32x2;
                                            else if (formatStr == "float32x3") va.format = WGPUVertexFormat_Float32x3;
                                            else if (formatStr == "float32x4") va.format = WGPUVertexFormat_Float32x4;
                                            else if (formatStr == "uint8x2") va.format = WGPUVertexFormat_Uint8x2;
                                            else if (formatStr == "uint8x4") va.format = WGPUVertexFormat_Uint8x4;
                                            else if (formatStr == "sint8x2") va.format = WGPUVertexFormat_Sint8x2;
                                            else if (formatStr == "sint8x4") va.format = WGPUVertexFormat_Sint8x4;
                                            else if (formatStr == "unorm8x2") va.format = WGPUVertexFormat_Unorm8x2;
                                            else if (formatStr == "unorm8x4") va.format = WGPUVertexFormat_Unorm8x4;
                                            else if (formatStr == "snorm8x2") va.format = WGPUVertexFormat_Snorm8x2;
                                            else if (formatStr == "snorm8x4") va.format = WGPUVertexFormat_Snorm8x4;
                                            else if (formatStr == "uint16x2") va.format = WGPUVertexFormat_Uint16x2;
                                            else if (formatStr == "uint16x4") va.format = WGPUVertexFormat_Uint16x4;
                                            else if (formatStr == "sint16x2") va.format = WGPUVertexFormat_Sint16x2;
                                            else if (formatStr == "sint16x4") va.format = WGPUVertexFormat_Sint16x4;
                                            else if (formatStr == "unorm16x2") va.format = WGPUVertexFormat_Unorm16x2;
                                            else if (formatStr == "unorm16x4") va.format = WGPUVertexFormat_Unorm16x4;
                                            else if (formatStr == "snorm16x2") va.format = WGPUVertexFormat_Snorm16x2;
                                            else if (formatStr == "snorm16x4") va.format = WGPUVertexFormat_Snorm16x4;
                                            else if (formatStr == "float16x2") va.format = WGPUVertexFormat_Float16x2;
                                            else if (formatStr == "float16x4") va.format = WGPUVertexFormat_Float16x4;
                                            else if (formatStr == "uint32") va.format = WGPUVertexFormat_Uint32;
                                            else if (formatStr == "uint32x2") va.format = WGPUVertexFormat_Uint32x2;
                                            else if (formatStr == "uint32x3") va.format = WGPUVertexFormat_Uint32x3;
                                            else if (formatStr == "uint32x4") va.format = WGPUVertexFormat_Uint32x4;
                                            else if (formatStr == "sint32") va.format = WGPUVertexFormat_Sint32;
                                            else if (formatStr == "sint32x2") va.format = WGPUVertexFormat_Sint32x2;
                                            else if (formatStr == "sint32x3") va.format = WGPUVertexFormat_Sint32x3;
                                            else if (formatStr == "sint32x4") va.format = WGPUVertexFormat_Sint32x4;
                                            else va.format = WGPUVertexFormat_Float32x3; // Default

                                            attributes.push_back(va);
                                        }

                                        allAttributes.push_back(attributes);
                                        layout.attributeCount = attributes.size();
                                        layout.attributes = allAttributes.back().data();
                                    }

                                    vertexBuffers.push_back(layout);
                                }

                                pipelineDesc.vertex.bufferCount = vertexBuffers.size();
                                pipelineDesc.vertex.buffers = vertexBuffers.data();
                            }

                            // Fragment state (only if fragment shader exists)
                            WGPUColorTargetState colorTarget = {};
                            WGPUFragmentState fragmentState = {};
                            std::vector<WGPUColorTargetState> colorTargets;
                            bool targetsExplicitlySpecified = false;
                            if (hasFragment && fsModule) {
                                // Parse targets from fragment descriptor
                                auto targetsProp = g_engine->getProperty(fragment, "targets");
                                if (!g_engine->isUndefined(targetsProp)) {
                                    if (!sequenceArray(targetsProp, targetsProp)) return g_engine->newUndefined();
                                    targetsExplicitlySpecified = true;  // Even if empty array
                                    auto targetsLen = g_engine->getProperty(targetsProp, "length");
                                    int targetCount = (int)g_engine->toNumber(targetsLen);
                                    for (int i = 0; i < targetCount; i++) {
                                        auto target = g_engine->getPropertyIndex(targetsProp, i);
                                        WGPUColorTargetState targetState = {};

                                        auto formatProp = g_engine->getProperty(target, "format");
                                        if (!g_engine->isUndefined(formatProp)) {
                                            std::string formatStr = g_engine->toString(formatProp);
                                            targetState.format = stringToFormat(formatStr);
                                            if (targetCount >= 5) {
                                                if (g_verboseLogging) std::cout << "[WebGPU] Pipeline target " << i << ": format=" << formatStr << " (enum=" << targetState.format << ")" << std::endl;
                                            }
                                        } else {
                                            targetState.format = g_surfaceFormat;
                                        }
                                        targetState.writeMask = WGPUColorWriteMask_All;
                                        auto writeMask = g_engine->getProperty(target, "writeMask");
                                        if (!g_engine->isUndefined(writeMask)) {
                                            const double mask = std::trunc(g_engine->toNumber(writeMask));
                                            if (!std::isfinite(mask) || mask < 0 || mask >= 4294967296.0) {
                                                g_engine->throwException("Color write mask is outside its unsigned integer range");
                                                return g_engine->newUndefined();
                                            }
                                            targetState.writeMask = static_cast<WGPUColorWriteMask>(mask);
                                        }

                                        // Parse blend state if provided
                                        auto blendProp = g_engine->getProperty(target, "blend");
                                        if (!g_engine->isUndefined(blendProp)) {
                                            // Store blend state in a persistent container
                                            static std::vector<std::unique_ptr<WGPUBlendState>> blendStates;
                                            auto blendState = std::make_unique<WGPUBlendState>();

                                            // Helper lambda to parse blend factor
                                            auto parseBlendFactor = [](const std::string& str) -> WGPUBlendFactor {
                                                if (str == "zero") return WGPUBlendFactor_Zero;
                                                if (str == "one") return WGPUBlendFactor_One;
                                                if (str == "src") return WGPUBlendFactor_Src;
                                                if (str == "one-minus-src") return WGPUBlendFactor_OneMinusSrc;
                                                if (str == "src-alpha") return WGPUBlendFactor_SrcAlpha;
                                                if (str == "one-minus-src-alpha") return WGPUBlendFactor_OneMinusSrcAlpha;
                                                if (str == "dst") return WGPUBlendFactor_Dst;
                                                if (str == "one-minus-dst") return WGPUBlendFactor_OneMinusDst;
                                                if (str == "dst-alpha") return WGPUBlendFactor_DstAlpha;
                                                if (str == "one-minus-dst-alpha") return WGPUBlendFactor_OneMinusDstAlpha;
                                                if (str == "src-alpha-saturated") return WGPUBlendFactor_SrcAlphaSaturated;
                                                if (str == "constant") return WGPUBlendFactor_Constant;
                                                if (str == "one-minus-constant") return WGPUBlendFactor_OneMinusConstant;
                                                return WGPUBlendFactor_One;  // Default
                                            };

                                            // Helper lambda to parse blend operation
                                            auto parseBlendOp = [](const std::string& str) -> WGPUBlendOperation {
                                                if (str == "add") return WGPUBlendOperation_Add;
                                                if (str == "subtract") return WGPUBlendOperation_Subtract;
                                                if (str == "reverse-subtract") return WGPUBlendOperation_ReverseSubtract;
                                                if (str == "min") return WGPUBlendOperation_Min;
                                                if (str == "max") return WGPUBlendOperation_Max;
                                                return WGPUBlendOperation_Add;  // Default
                                            };

                                            // Parse color blend component
                                            auto colorProp = g_engine->getProperty(blendProp, "color");
                                            if (!g_engine->isUndefined(colorProp)) {
                                                auto srcFactor = g_engine->getProperty(colorProp, "srcFactor");
                                                auto dstFactor = g_engine->getProperty(colorProp, "dstFactor");
                                                auto operation = g_engine->getProperty(colorProp, "operation");
                                                if (!g_engine->isUndefined(srcFactor))
                                                    blendState->color.srcFactor = parseBlendFactor(g_engine->toString(srcFactor));
                                                else
                                                    blendState->color.srcFactor = WGPUBlendFactor_One;
                                                if (!g_engine->isUndefined(dstFactor))
                                                    blendState->color.dstFactor = parseBlendFactor(g_engine->toString(dstFactor));
                                                else
                                                    blendState->color.dstFactor = WGPUBlendFactor_Zero;
                                                if (!g_engine->isUndefined(operation))
                                                    blendState->color.operation = parseBlendOp(g_engine->toString(operation));
                                                else
                                                    blendState->color.operation = WGPUBlendOperation_Add;
                                            } else {
                                                // Default color blend (no blending)
                                                blendState->color.srcFactor = WGPUBlendFactor_One;
                                                blendState->color.dstFactor = WGPUBlendFactor_Zero;
                                                blendState->color.operation = WGPUBlendOperation_Add;
                                            }

                                            // Parse alpha blend component
                                            auto alphaProp = g_engine->getProperty(blendProp, "alpha");
                                            if (!g_engine->isUndefined(alphaProp)) {
                                                auto srcFactor = g_engine->getProperty(alphaProp, "srcFactor");
                                                auto dstFactor = g_engine->getProperty(alphaProp, "dstFactor");
                                                auto operation = g_engine->getProperty(alphaProp, "operation");
                                                if (!g_engine->isUndefined(srcFactor))
                                                    blendState->alpha.srcFactor = parseBlendFactor(g_engine->toString(srcFactor));
                                                else
                                                    blendState->alpha.srcFactor = WGPUBlendFactor_One;
                                                if (!g_engine->isUndefined(dstFactor))
                                                    blendState->alpha.dstFactor = parseBlendFactor(g_engine->toString(dstFactor));
                                                else
                                                    blendState->alpha.dstFactor = WGPUBlendFactor_Zero;
                                                if (!g_engine->isUndefined(operation))
                                                    blendState->alpha.operation = parseBlendOp(g_engine->toString(operation));
                                                else
                                                    blendState->alpha.operation = WGPUBlendOperation_Add;
                                            } else {
                                                // Default alpha blend (no blending)
                                                blendState->alpha.srcFactor = WGPUBlendFactor_One;
                                                blendState->alpha.dstFactor = WGPUBlendFactor_Zero;
                                                blendState->alpha.operation = WGPUBlendOperation_Add;
                                            }

                                            targetState.blend = blendState.get();
                                            blendStates.push_back(std::move(blendState));

                                            if (g_verboseLogging) std::cout << "[WebGPU] Pipeline target " << i << " has blend state" << std::endl;
                                        }

                                        colorTargets.push_back(targetState);
                                    }
                                }
                                // Only add default target if targets wasn't explicitly specified
                                // If targets: [] was specified, don't add any (depth-only pass)
                                if (colorTargets.empty() && !targetsExplicitlySpecified) {
                                    // Default single target only when targets is not specified at all
                                    colorTarget.format = g_surfaceFormat;
                                    colorTarget.writeMask = WGPUColorWriteMask_All;
                                    colorTargets.push_back(colorTarget);
                                }

                                fragmentState.module = fsModule;
                                WGPU_SET_ENTRY_POINT(fragmentState, fragmentEntry.c_str());
                                fragmentState.targetCount = colorTargets.size();
                                fragmentState.targets = colorTargets.data();
                                pipelineDesc.fragment = &fragmentState;
                                if (g_verboseLogging) std::cout << "[WebGPU] Render pipeline with " << colorTargets.size() << " color targets" << std::endl;
                            } else {
                                // Depth-only pipeline - no fragment state
                                pipelineDesc.fragment = nullptr;
                            }

                            // Primitive state
                            pipelineDesc.primitive.topology = WGPUPrimitiveTopology_TriangleList;
                            pipelineDesc.primitive.stripIndexFormat = WGPUIndexFormat_Undefined;
                            pipelineDesc.primitive.frontFace = WGPUFrontFace_CCW;
                            pipelineDesc.primitive.cullMode = WGPUCullMode_None;

                            // Parse primitive state if provided
                            auto primitiveProp = g_engine->getProperty(descriptor, "primitive");
                            if (!g_engine->isUndefined(primitiveProp)) {
                                auto topologyProp = g_engine->getProperty(primitiveProp, "topology");
                                if (!g_engine->isUndefined(topologyProp)) {
                                    std::string topologyStr = g_engine->toString(topologyProp);
                                    if (topologyStr == "point-list") pipelineDesc.primitive.topology = WGPUPrimitiveTopology_PointList;
                                    else if (topologyStr == "line-list") pipelineDesc.primitive.topology = WGPUPrimitiveTopology_LineList;
                                    else if (topologyStr == "line-strip") pipelineDesc.primitive.topology = WGPUPrimitiveTopology_LineStrip;
                                    else if (topologyStr == "triangle-list") pipelineDesc.primitive.topology = WGPUPrimitiveTopology_TriangleList;
                                    else if (topologyStr == "triangle-strip") pipelineDesc.primitive.topology = WGPUPrimitiveTopology_TriangleStrip;
                                }
                                auto cullModeProp = g_engine->getProperty(primitiveProp, "cullMode");
                                if (!g_engine->isUndefined(cullModeProp)) {
                                    std::string cullModeStr = g_engine->toString(cullModeProp);
                                    if (cullModeStr == "none") pipelineDesc.primitive.cullMode = WGPUCullMode_None;
                                    else if (cullModeStr == "front") pipelineDesc.primitive.cullMode = WGPUCullMode_Front;
                                    else if (cullModeStr == "back") pipelineDesc.primitive.cullMode = WGPUCullMode_Back;
                                }
                                auto frontFaceProp = g_engine->getProperty(primitiveProp, "frontFace");
                                if (!g_engine->isUndefined(frontFaceProp)) {
                                    std::string frontFaceStr = g_engine->toString(frontFaceProp);
                                    if (frontFaceStr == "ccw") pipelineDesc.primitive.frontFace = WGPUFrontFace_CCW;
                                    else if (frontFaceStr == "cw") pipelineDesc.primitive.frontFace = WGPUFrontFace_CW;
                                }
                            }

                            // Depth stencil state
                            WGPUDepthStencilState depthStencilState = {};
                            bool hasDepthStencil = false;

                            auto depthStencilProp = g_engine->getProperty(descriptor, "depthStencil");
                            if (!g_engine->isUndefined(depthStencilProp)) {
                                hasDepthStencil = true;

                                auto formatProp = g_engine->getProperty(depthStencilProp, "format");
                                if (!g_engine->isUndefined(formatProp)) {
                                    depthStencilState.format = stringToFormat(g_engine->toString(formatProp));
                                } else {
                                    depthStencilState.format = WGPUTextureFormat_Depth24Plus;
                                }

                                auto depthWriteEnabledProp = g_engine->getProperty(depthStencilProp, "depthWriteEnabled");
                                depthStencilState.depthWriteEnabled = g_engine->isUndefined(depthWriteEnabledProp)
                                    ? WGPU_OPTIONAL_BOOL_TRUE
                                    : (g_engine->toBoolean(depthWriteEnabledProp) ? WGPU_OPTIONAL_BOOL_TRUE : WGPU_OPTIONAL_BOOL_FALSE);

                                auto depthCompareProp = g_engine->getProperty(depthStencilProp, "depthCompare");
                                if (!g_engine->isUndefined(depthCompareProp)) {
                                    std::string compareStr = g_engine->toString(depthCompareProp);
                                    if (compareStr == "never") depthStencilState.depthCompare = WGPUCompareFunction_Never;
                                    else if (compareStr == "less") depthStencilState.depthCompare = WGPUCompareFunction_Less;
                                    else if (compareStr == "less-equal") depthStencilState.depthCompare = WGPUCompareFunction_LessEqual;
                                    else if (compareStr == "greater") depthStencilState.depthCompare = WGPUCompareFunction_Greater;
                                    else if (compareStr == "greater-equal") depthStencilState.depthCompare = WGPUCompareFunction_GreaterEqual;
                                    else if (compareStr == "equal") depthStencilState.depthCompare = WGPUCompareFunction_Equal;
                                    else if (compareStr == "not-equal") depthStencilState.depthCompare = WGPUCompareFunction_NotEqual;
                                    else if (compareStr == "always") depthStencilState.depthCompare = WGPUCompareFunction_Always;
                                } else {
                                    depthStencilState.depthCompare = WGPUCompareFunction_Less;
                                }

                                // Default stencil operations
                                depthStencilState.stencilFront.compare = WGPUCompareFunction_Always;
                                depthStencilState.stencilFront.failOp = WGPUStencilOperation_Keep;
                                depthStencilState.stencilFront.depthFailOp = WGPUStencilOperation_Keep;
                                depthStencilState.stencilFront.passOp = WGPUStencilOperation_Keep;
                                depthStencilState.stencilBack = depthStencilState.stencilFront;
                                depthStencilState.stencilReadMask = 0xFFFFFFFF;
                                depthStencilState.stencilWriteMask = 0xFFFFFFFF;

                                pipelineDesc.depthStencil = &depthStencilState;
                            }

                            // Multisample state - parse from descriptor or use defaults
                            pipelineDesc.multisample.count = 1;
                            pipelineDesc.multisample.mask = 0xFFFFFFFF;
                            pipelineDesc.multisample.alphaToCoverageEnabled = false;

                            auto multisampleProp = g_engine->getProperty(descriptor, "multisample");
                            if (!g_engine->isUndefined(multisampleProp)) {
                                auto countProp = g_engine->getProperty(multisampleProp, "count");
                                if (!g_engine->isUndefined(countProp)) {
                                    pipelineDesc.multisample.count = (uint32_t)g_engine->toNumber(countProp);
                                }

                                auto maskProp = g_engine->getProperty(multisampleProp, "mask");
                                if (!g_engine->isUndefined(maskProp)) {
                                    pipelineDesc.multisample.mask = (uint32_t)g_engine->toNumber(maskProp);
                                }

                                auto alphaToCoverageProp = g_engine->getProperty(multisampleProp, "alphaToCoverageEnabled");
                                if (!g_engine->isUndefined(alphaToCoverageProp)) {
                                    pipelineDesc.multisample.alphaToCoverageEnabled = g_engine->toBoolean(alphaToCoverageProp);
                                }

                                if (g_verboseLogging) {
                                    std::cout << "[WebGPU] Render pipeline multisample: count=" << pipelineDesc.multisample.count
                                              << ", mask=" << pipelineDesc.multisample.mask << std::endl;
                                }
                            }

                            // Create pipeline
                            WGPURenderPipeline pipeline = wgpuDeviceCreateRenderPipeline(g_device, &pipelineDesc);
                            if (!pipeline) {
                                g_engine->throwException("Failed to create render pipeline");
                                return g_engine->newUndefined();
                            }

                            // Register pipeline for getBindGroupLayout
                            uint64_t pipelineId = g_nextRenderPipelineId++;
                            g_renderPipelineRegistry[pipelineId] = pipeline;

                            auto jsPipeline = g_engine->newObject();
                            manageResource(jsPipeline, pipeline, ResourceKind::RenderPipeline, pipelineId);
                            g_engine->setProperty(jsPipeline, "_pipelineId", g_engine->newNumber((double)pipelineId));
                            g_engine->setProperty(jsPipeline, "_type", g_engine->newString("renderPipeline"));

                            // Add getBindGroupLayout method using captured pipelineId
                            g_engine->setProperty(jsPipeline, "getBindGroupLayout",
                                g_engine->newFunction("getBindGroupLayout", [pipelineId](void* ctx, const std::vector<js::JSValueHandle>& args) {
                                    auto it = g_renderPipelineRegistry.find(pipelineId);
                                    if (it == g_renderPipelineRegistry.end() || !it->second) {
                                        std::cerr << "[WebGPU] getBindGroupLayout: Render pipeline not found" << std::endl;
                                        return g_engine->newUndefined();
                                    }

                                    uint32_t groupIndex = args.empty() ? 0 : (uint32_t)g_engine->toNumber(args[0]);
                                    WGPUBindGroupLayout layout = wgpuRenderPipelineGetBindGroupLayout(it->second, groupIndex);

                                    if (!layout) {
                                        std::cerr << "[WebGPU] getBindGroupLayout: Failed to get layout for group " << groupIndex << std::endl;
                                        return g_engine->newUndefined();
                                    }

                                    auto jsLayout = g_engine->newObject();
                                    manageResource(jsLayout, layout, ResourceKind::BindGroupLayout);
                                    g_engine->setProperty(jsLayout, "_type", g_engine->newString("bindGroupLayout"));

                                    return jsLayout;
                                })
                            );

                            if (g_verboseLogging) std::cout << "[WebGPU] Render pipeline created (id=" << pipelineId << ")" << std::endl;
                            return jsPipeline;
                        })
                    );

                    // device.createComputePipeline(descriptor)
                    g_engine->setProperty(device, "createComputePipeline",
                        g_engine->newFunction("createComputePipeline", [](void* ctx, const std::vector<js::JSValueHandle>& args) {
                            if (args.empty()) {
                                g_engine->throwException("createComputePipeline requires a descriptor");
                                return g_engine->newUndefined();
                            }

                            auto descriptor = args[0];

                            // Get layout
                            auto layoutProp = g_engine->getProperty(descriptor, "layout");
                            WGPUPipelineLayout layout = nullptr;
                            bool isAutoLayout = false;
                            if (!g_engine->isUndefined(layoutProp) && !g_engine->isString(layoutProp)) {
                                layout = (WGPUPipelineLayout)resourceData(layoutProp);
                            } else if (g_engine->isString(layoutProp)) {
                                std::string layoutStr = g_engine->toString(layoutProp);
                                if (layoutStr == "auto") {
                                    isAutoLayout = true;
                                    if (g_verboseLogging) std::cout << "[WebGPU] Using 'auto' layout for compute pipeline" << std::endl;
                                    std::cout.flush();
                                }
                            }

                            // Get compute stage
                            auto computeProp = g_engine->getProperty(descriptor, "compute");
                            auto moduleProp = g_engine->getProperty(computeProp, "module");
                            WGPUShaderModule module = (WGPUShaderModule)resourceData(moduleProp);

                            // Entry point (default "main")
                            std::string entryPoint = "main";
                            auto entryPointProp = g_engine->getProperty(computeProp, "entryPoint");
                            if (!g_engine->isUndefined(entryPointProp)) {
                                entryPoint = g_engine->toString(entryPointProp);
                            }

                            // Create pipeline
                            WGPUComputePipelineDescriptor pipelineDesc = {};
                            pipelineDesc.layout = layout;
                            pipelineDesc.compute.module = module;
                            WGPU_SET_ENTRY_POINT(pipelineDesc.compute, entryPoint.c_str());

                            WGPUComputePipeline pipeline = wgpuDeviceCreateComputePipeline(g_device, &pipelineDesc);
                            if (!pipeline) {
                                g_engine->throwException("Failed to create compute pipeline");
                                return g_engine->newUndefined();
                            }

                            // Register pipeline for getBindGroupLayout
                            uint64_t pipelineId = g_nextComputePipelineId++;
                            g_computePipelineRegistry[pipelineId] = pipeline;

                            auto jsPipeline = g_engine->newObject();
                            manageResource(jsPipeline, pipeline, ResourceKind::ComputePipeline, pipelineId);
                            g_engine->setProperty(jsPipeline, "_pipelineId", g_engine->newNumber((double)pipelineId));
                            g_engine->setProperty(jsPipeline, "_type", g_engine->newString("computePipeline"));

                            // Add getBindGroupLayout method using captured pipelineId
                            g_engine->setProperty(jsPipeline, "getBindGroupLayout",
                                g_engine->newFunction("getBindGroupLayout", [pipelineId](void* ctx, const std::vector<js::JSValueHandle>& args) {
                                    auto it = g_computePipelineRegistry.find(pipelineId);
                                    if (it == g_computePipelineRegistry.end() || !it->second) {
                                        std::cerr << "[WebGPU] getBindGroupLayout: Compute pipeline not found" << std::endl;
                                        return g_engine->newUndefined();
                                    }

                                    uint32_t groupIndex = args.empty() ? 0 : (uint32_t)g_engine->toNumber(args[0]);
                                    WGPUBindGroupLayout layout = wgpuComputePipelineGetBindGroupLayout(it->second, groupIndex);

                                    if (!layout) {
                                        std::cerr << "[WebGPU] getBindGroupLayout: Failed to get layout for group " << groupIndex << std::endl;
                                        return g_engine->newUndefined();
                                    }

                                    auto jsLayout = g_engine->newObject();
                                    manageResource(jsLayout, layout, ResourceKind::BindGroupLayout);
                                    g_engine->setProperty(jsLayout, "_type", g_engine->newString("bindGroupLayout"));

                                    return jsLayout;
                                })
                            );

                            if (g_verboseLogging) std::cout << "[WebGPU] Compute pipeline created (id=" << pipelineId << ")" << std::endl;
                            return jsPipeline;
                        })
                    );

                    // device.createCommandEncoder(descriptor?)
                    g_engine->setProperty(device, "createCommandEncoder",
                        g_engine->newFunction("createCommandEncoder", [](void* ctx, const std::vector<js::JSValueHandle>& args) {
                            WGPUCommandEncoderDescriptor desc = {};
                            WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(g_device, &desc);

                            // Store in global for use by beginRenderPass
                            // Note: Multiple encoders are supported via per-encoder render pass tracking
                            g_jsCommandEncoder = encoder;

                            // Suspend frame tracking while creating encoder wrapper
                            // This prevents the wrapper's methods from being garbage collected at frame end
                            g_engine->suspendFrameTracking();

                            auto jsEncoder = g_engine->newObject();
                            const auto encoderResourceId = manageResource(jsEncoder, encoder, ResourceKind::CommandEncoder);

                            // Capture encoder pointer for use in closures
                            WGPUCommandEncoder capturedEncoder = encoder;

                            // encoder.beginRenderPass(descriptor)
                            g_engine->setProperty(jsEncoder, "beginRenderPass",
                                g_engine->newFunction("beginRenderPass", [capturedEncoder, encoderResourceId](void* ctx, const std::vector<js::JSValueHandle>& args) {
                                            auto activeResource = resourceHandle<WGPUCommandEncoder>(encoderResourceId);
                                            if (!activeResource) return g_engine->newUndefined();
                                            g_jsCommandEncoder = activeResource;
                                    if (args.empty()) {
                                        g_engine->throwException("beginRenderPass requires a descriptor");
                                        return g_engine->newUndefined();
                                    }

                                    // Use the captured encoder for this specific command encoder
                                    WGPUCommandEncoder encoderToUse = capturedEncoder;
                                    if (!encoderToUse) {
                                        g_engine->throwException("Command encoder not available");
                                        return g_engine->newUndefined();
                                    }

                                    auto descriptor = args[0];
                                    auto colorAttachments = g_engine->getProperty(descriptor, "colorAttachments");
                                    if (!sequenceArray(colorAttachments, colorAttachments)) return g_engine->newUndefined();

                                    // Parse all color attachments (deferred renderer uses multiple)
                                    auto attachmentsLengthProp = g_engine->getProperty(colorAttachments, "length");
                                    int numAttachments = g_engine->isUndefined(attachmentsLengthProp) ? 0 : (int)g_engine->toNumber(attachmentsLengthProp);
                                    std::vector<WGPURenderPassColorAttachment> colorAttachmentList;
                                    colorAttachmentList.reserve(numAttachments);

                                    double firstR = 0, firstG = 0, firstB = 0, firstA = 1;

                                    for (int i = 0; i < numAttachments; i++) {
                                        auto attachment = g_engine->getPropertyIndex(colorAttachments, i);
                                        auto viewHandle = g_engine->getProperty(attachment, "view");
                                        WGPUTextureView view = (WGPUTextureView)resourceData(viewHandle);
                                        if (!view) {
                                            g_engine->throwException("Render attachment view is invalid or expired");
                                            return g_engine->newUndefined();
                                        }
                                        auto resolveTargetHandle = g_engine->getProperty(attachment, "resolveTarget");
                                        WGPUTextureView resolveTarget = g_engine->isUndefined(resolveTargetHandle)
                                            ? nullptr
                                            : (WGPUTextureView)resourceData(resolveTargetHandle);
                                        if (!g_engine->isUndefined(resolveTargetHandle) && !g_engine->isNull(resolveTargetHandle) && !resolveTarget)
                                            return g_engine->newUndefined();

                                        // Debug: Log first color attachment for comparison with g_currentTextureView
                                        if (i == 0) {
                                            if (g_verboseLogging) {
                                                std::cout << "[WebGPU] Render pass attachment[0]: view=" << (void*)view
                                                          << ", g_currentTextureView=" << (void*)g_currentTextureView
                                                          << ", matches=" << (view == g_currentTextureView ? "YES" : "NO") << std::endl;
                                            }

                                            // Track if this render pass uses the surface texture
                                            if (isCanvasView(view) || isCanvasView(resolveTarget)) {
                                                g_surfaceRenderEncoder = encoderToUse;
                                                g_surfaceRenderPassEnded = false;
                                            }
                                        }

                                        // Debug: Log GBuffer pass attachments
                                        if (numAttachments >= 5 && i == 0) {
                                            if (g_verboseLogging) std::cout << "[WebGPU] GBuffer pass - 5 attachments, view[0]=" << (void*)view << std::endl;
                                        }
                                        if (!view && numAttachments >= 5) {
                                            std::cerr << "[WebGPU] ERROR: GBuffer attachment " << i << " has null view!" << std::endl;
                                        }

                                        // Parse loadOp (default 'clear')
                                        WGPULoadOp loadOp = WGPULoadOp_Clear;
                                        auto loadOpProp = g_engine->getProperty(attachment, "loadOp");
                                        if (!g_engine->isUndefined(loadOpProp)) {
                                            std::string loadOpStr = g_engine->toString(loadOpProp);
                                            if (loadOpStr == "load") loadOp = WGPULoadOp_Load;
                                        }

                                        // Parse storeOp (default 'store')
                                        WGPUStoreOp storeOp = WGPUStoreOp_Store;
                                        auto storeOpProp = g_engine->getProperty(attachment, "storeOp");
                                        if (!g_engine->isUndefined(storeOpProp)) {
                                            std::string storeOpStr = g_engine->toString(storeOpProp);
                                            if (storeOpStr == "discard") storeOp = WGPUStoreOp_Discard;
                                        }

                                        // Parse clearValue only if loadOp is 'clear'
                                        double r = 0, g = 0, b = 0, a = 1;
                                        if (loadOp == WGPULoadOp_Clear) {
                                            auto clearValue = g_engine->getProperty(attachment, "clearValue");
                                            if (!g_engine->isUndefined(clearValue)) {
                                                // Check if it's an array [r, g, b, a] or object {r, g, b, a}
                                                if (g_engine->isArray(clearValue)) {
                                                    r = g_engine->toNumber(g_engine->getPropertyIndex(clearValue, 0));
                                                    g = g_engine->toNumber(g_engine->getPropertyIndex(clearValue, 1));
                                                    b = g_engine->toNumber(g_engine->getPropertyIndex(clearValue, 2));
                                                    a = g_engine->toNumber(g_engine->getPropertyIndex(clearValue, 3));
                                                } else {
                                                    r = g_engine->toNumber(g_engine->getProperty(clearValue, "r"));
                                                    g = g_engine->toNumber(g_engine->getProperty(clearValue, "g"));
                                                    b = g_engine->toNumber(g_engine->getProperty(clearValue, "b"));
                                                    a = g_engine->toNumber(g_engine->getProperty(clearValue, "a"));
                                                }
                                            }
                                        }

                                        if (i == 0) {
                                            firstR = r; firstG = g; firstB = b; firstA = a;
                                        }

                                        WGPURenderPassColorAttachment colorAttachment = {};
                                        colorAttachment.view = view;
                                        colorAttachment.resolveTarget = resolveTarget;
                                        colorAttachment.loadOp = loadOp;
                                        colorAttachment.storeOp = storeOp;
                                        colorAttachment.clearValue = {r, g, b, a};
                                        colorAttachment.depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;
                                        colorAttachmentList.push_back(colorAttachment);
                                    }

                                    WGPURenderPassDescriptor renderPassDesc = {};
                                    renderPassDesc.colorAttachmentCount = colorAttachmentList.size();
                                    renderPassDesc.colorAttachments = colorAttachmentList.data();

                                    // Parse depth stencil attachment if present
                                    WGPURenderPassDepthStencilAttachment depthStencilAttachment = {};
                                    auto depthStencilProp = g_engine->getProperty(descriptor, "depthStencilAttachment");
                                    if (!g_engine->isUndefined(depthStencilProp)) {
                                        auto depthViewHandle = g_engine->getProperty(depthStencilProp, "view");
                                        WGPUTextureView depthView = (WGPUTextureView)resourceData(depthViewHandle);
                                        if (!depthView) {
                                            g_engine->throwException("Depth attachment view is invalid or expired");
                                            return g_engine->newUndefined();
                                        }
                                        depthStencilAttachment.view = depthView;

                                        // Preserve optional operations for native attachment/format validation.
                                        auto depthClearValueProp = g_engine->getProperty(depthStencilProp, "depthClearValue");
                                        depthStencilAttachment.depthClearValue = g_engine->isUndefined(depthClearValueProp)
                                            ? WGPU_DEPTH_CLEAR_VALUE_UNDEFINED : (float)g_engine->toNumber(depthClearValueProp);
                                        if (!renderPassLoadOp(g_engine->getProperty(depthStencilProp, "depthLoadOp"), depthStencilAttachment.depthLoadOp) ||
                                            !renderPassStoreOp(g_engine->getProperty(depthStencilProp, "depthStoreOp"), depthStencilAttachment.depthStoreOp) ||
                                            !renderPassLoadOp(g_engine->getProperty(depthStencilProp, "stencilLoadOp"), depthStencilAttachment.stencilLoadOp) ||
                                            !renderPassStoreOp(g_engine->getProperty(depthStencilProp, "stencilStoreOp"), depthStencilAttachment.stencilStoreOp))
                                            return g_engine->newUndefined();
                                        auto depthReadOnly = g_engine->getProperty(depthStencilProp, "depthReadOnly");
                                        auto stencilReadOnly = g_engine->getProperty(depthStencilProp, "stencilReadOnly");
                                        depthStencilAttachment.depthReadOnly = !g_engine->isUndefined(depthReadOnly) && g_engine->toBoolean(depthReadOnly);
                                        depthStencilAttachment.stencilReadOnly = !g_engine->isUndefined(stencilReadOnly) && g_engine->toBoolean(stencilReadOnly);
                                        auto stencilClear = g_engine->getProperty(depthStencilProp, "stencilClearValue");
                                        uint64_t stencilClearValue;
                                        if (!copyUnsigned(stencilClear, 0, false, stencilClearValue)) return g_engine->newUndefined();
                                        depthStencilAttachment.stencilClearValue = static_cast<uint32_t>(stencilClearValue);

                                        renderPassDesc.depthStencilAttachment = &depthStencilAttachment;
                                        if (g_verboseLogging) std::cout << "[WebGPU] Render pass with depth attachment, clear=" << depthStencilAttachment.depthClearValue << std::endl;
                                    }

                                    // Begin render pass on the captured encoder (not the global)
                                    WGPURenderPassEncoder renderPass = wgpuCommandEncoderBeginRenderPass(encoderToUse, &renderPassDesc);

                                    // Store in per-encoder map (fixes issue with multiple encoders)
                                    g_encoderRenderPassMap[encoderToUse] = renderPass;

                                    // Also set global for backwards compatibility with render pass methods
                                    g_jsRenderPass = renderPass;

                                    if (g_verboseLogging) std::cout << "[WebGPU] Render pass started (" << numAttachments << " attachments), clear: (" << firstR << "," << firstG << "," << firstB << "," << firstA << ")" << std::endl;

                                    // Suspend frame tracking while creating render pass wrapper
                                    g_engine->suspendFrameTracking();

                                    auto jsRenderPass = g_engine->newObject();
                                    const auto passResourceId = manageResource(jsRenderPass, renderPass, ResourceKind::RenderPass);

                                    // renderPass.setPipeline(pipeline)
                                    g_engine->setProperty(jsRenderPass, "setPipeline",
                                        g_engine->newFunction("setPipeline", [passResourceId](void* ctx, const std::vector<js::JSValueHandle>& args) {
                                            auto activeResource = resourceHandle<WGPURenderPassEncoder>(passResourceId);
                                            if (!activeResource) return g_engine->newUndefined();
                                            g_jsRenderPass = activeResource;
                                            if (args.empty()) return g_engine->newUndefined();

                                            WGPURenderPipeline pipeline = (WGPURenderPipeline)resourceData(args[0]);
                                            if (g_jsRenderPass && pipeline) {
                                                wgpuRenderPassEncoderSetPipeline(g_jsRenderPass, pipeline);
                                                if (g_verboseLogging) std::cout << "[WebGPU] Pipeline set" << std::endl;
                                            }

                                            return g_engine->newUndefined();
                                        })
                                    );

                                    // renderPass.setBindGroup(index, bindGroup, dynamicOffsets?)
                                    g_engine->setProperty(jsRenderPass, "setBindGroup",
                                        g_engine->newFunction("setBindGroup", [passResourceId](void* ctx, const std::vector<js::JSValueHandle>& args) {
                                            auto activeResource = resourceHandle<WGPURenderPassEncoder>(passResourceId);
                                            if (!activeResource) return g_engine->newUndefined();
                                            g_jsRenderPass = activeResource;
                                            if (args.size() < 2) {
                                                g_engine->throwException("setBindGroup requires index and bindGroup");
                                                return g_engine->newUndefined();
                                            }

                                            uint32_t groupIndex = (uint32_t)g_engine->toNumber(args[0]);
                                            WGPUBindGroup bindGroup = g_engine->isNull(args[1]) || g_engine->isUndefined(args[1]) ?
                                                nullptr : static_cast<WGPUBindGroup>(resourceData(args[1]));
                                            std::vector<uint32_t> offsets;
                                            if (!dynamicOffsets(args, offsets)) return g_engine->newUndefined();
                                            if (g_jsRenderPass) {
                                                wgpuRenderPassEncoderSetBindGroup(g_jsRenderPass, groupIndex, bindGroup, offsets.size(), offsets.data());
                                                if (g_verboseLogging) std::cout << "[WebGPU] Set bind group at index " << groupIndex << std::endl;
                                            }

                                            return g_engine->newUndefined();
                                        })
                                    );

                                    // renderPass.draw(vertexCount, instanceCount?, firstVertex?, firstInstance?)
                                    g_engine->setProperty(jsRenderPass, "draw",
                                        g_engine->newFunction("draw", [passResourceId](void* ctx, const std::vector<js::JSValueHandle>& args) {
                                            auto activeResource = resourceHandle<WGPURenderPassEncoder>(passResourceId);
                                            if (!activeResource) return g_engine->newUndefined();
                                            g_jsRenderPass = activeResource;
                                            if (args.empty()) return g_engine->newUndefined();

                                            uint32_t vertexCount = (uint32_t)g_engine->toNumber(args[0]);
                                            uint32_t instanceCount = args.size() > 1 ? (uint32_t)g_engine->toNumber(args[1]) : 1;
                                            uint32_t firstVertex = args.size() > 2 ? (uint32_t)g_engine->toNumber(args[2]) : 0;
                                            uint32_t firstInstance = args.size() > 3 ? (uint32_t)g_engine->toNumber(args[3]) : 0;

                                            if (g_jsRenderPass) {
                                                wgpuRenderPassEncoderDraw(g_jsRenderPass, vertexCount, instanceCount, firstVertex, firstInstance);
                                                if (g_verboseLogging) std::cout << "[WebGPU] Draw: " << vertexCount << " vertices" << std::endl;
                                            }

                                            return g_engine->newUndefined();
                                        })
                                    );

                                    // renderPass.setVertexBuffer(slot, buffer, offset?, size?)
                                    g_engine->setProperty(jsRenderPass, "setVertexBuffer",
                                        g_engine->newFunction("setVertexBuffer", [passResourceId](void* ctx, const std::vector<js::JSValueHandle>& args) {
                                            auto activeResource = resourceHandle<WGPURenderPassEncoder>(passResourceId);
                                            if (!activeResource) return g_engine->newUndefined();
                                            g_jsRenderPass = activeResource;
                                            if (args.size() < 2) return g_engine->newUndefined();

                                            uint32_t slot = (uint32_t)g_engine->toNumber(args[0]);
                                            WGPUBuffer buffer = (WGPUBuffer)resourceData(args[1]);
                                            uint64_t offset = args.size() > 2 ? (uint64_t)g_engine->toNumber(args[2]) : 0;
                                            uint64_t size = args.size() > 3 ? (uint64_t)g_engine->toNumber(args[3]) : WGPU_WHOLE_SIZE;

                                            if (g_jsRenderPass && buffer) {
                                                wgpuRenderPassEncoderSetVertexBuffer(g_jsRenderPass, slot, buffer, offset, size);
                                                if (g_verboseLogging) std::cout << "[WebGPU] Set vertex buffer at slot " << slot << std::endl;
                                            }

                                            return g_engine->newUndefined();
                                        })
                                    );

                                    // renderPass.setIndexBuffer(buffer, format, offset?, size?)
                                    g_engine->setProperty(jsRenderPass, "setIndexBuffer",
                                        g_engine->newFunction("setIndexBuffer", [passResourceId](void* ctx, const std::vector<js::JSValueHandle>& args) {
                                            auto activeResource = resourceHandle<WGPURenderPassEncoder>(passResourceId);
                                            if (!activeResource) return g_engine->newUndefined();
                                            g_jsRenderPass = activeResource;
                                            if (args.size() < 2) return g_engine->newUndefined();

                                            WGPUBuffer buffer = (WGPUBuffer)resourceData(args[0]);
                                            std::string formatStr = g_engine->toString(args[1]);
                                            uint64_t offset = args.size() > 2 ? (uint64_t)g_engine->toNumber(args[2]) : 0;
                                            uint64_t size = args.size() > 3 ? (uint64_t)g_engine->toNumber(args[3]) : WGPU_WHOLE_SIZE;

                                            WGPUIndexFormat format = WGPUIndexFormat_Uint16;
                                            if (formatStr == "uint32") format = WGPUIndexFormat_Uint32;
                                            else if (formatStr == "uint16") format = WGPUIndexFormat_Uint16;

                                            if (g_jsRenderPass && buffer) {
                                                wgpuRenderPassEncoderSetIndexBuffer(g_jsRenderPass, buffer, format, offset, size);
                                                if (g_verboseLogging) std::cout << "[WebGPU] Set index buffer, format: " << formatStr << std::endl;
                                            }

                                            return g_engine->newUndefined();
                                        })
                                    );

                                    // renderPass.drawIndexed(indexCount, instanceCount?, firstIndex?, baseVertex?, firstInstance?)
                                    g_engine->setProperty(jsRenderPass, "drawIndexed",
                                        g_engine->newFunction("drawIndexed", [passResourceId](void* ctx, const std::vector<js::JSValueHandle>& args) {
                                            auto activeResource = resourceHandle<WGPURenderPassEncoder>(passResourceId);
                                            if (!activeResource) return g_engine->newUndefined();
                                            g_jsRenderPass = activeResource;
                                            if (args.empty()) return g_engine->newUndefined();

                                            uint32_t indexCount = (uint32_t)g_engine->toNumber(args[0]);
                                            uint32_t instanceCount = args.size() > 1 ? (uint32_t)g_engine->toNumber(args[1]) : 1;
                                            uint32_t firstIndex = args.size() > 2 ? (uint32_t)g_engine->toNumber(args[2]) : 0;
                                            int32_t baseVertex = args.size() > 3 ? (int32_t)g_engine->toNumber(args[3]) : 0;
                                            uint32_t firstInstance = args.size() > 4 ? (uint32_t)g_engine->toNumber(args[4]) : 0;

                                            if (g_jsRenderPass) {
                                                wgpuRenderPassEncoderDrawIndexed(g_jsRenderPass, indexCount, instanceCount, firstIndex, baseVertex, firstInstance);
                                                if (g_verboseLogging) std::cout << "[WebGPU] DrawIndexed: " << indexCount << " indices, firstInstance=" << firstInstance << std::endl;
                                            }

                                            return g_engine->newUndefined();
                                        })
                                    );

                                    // renderPass.drawIndirect(indirectBuffer, indirectOffset)
                                    g_engine->setProperty(jsRenderPass, "drawIndirect",
                                        g_engine->newFunction("drawIndirect", [passResourceId](void* ctx, const std::vector<js::JSValueHandle>& args) {
                                            auto activeResource = resourceHandle<WGPURenderPassEncoder>(passResourceId);
                                            if (!activeResource) return g_engine->newUndefined();
                                            g_jsRenderPass = activeResource;
                                            if (args.size() < 2) return g_engine->newUndefined();

                                            WGPUBuffer indirectBuffer = (WGPUBuffer)resourceData(args[0]);
                                            uint64_t indirectOffset = (uint64_t)g_engine->toNumber(args[1]);

                                            if (g_jsRenderPass && indirectBuffer) {
                                                wgpuRenderPassEncoderDrawIndirect(g_jsRenderPass, indirectBuffer, indirectOffset);
                                                if (g_verboseLogging) std::cout << "[WebGPU] DrawIndirect at offset " << indirectOffset << std::endl;
                                            }

                                            return g_engine->newUndefined();
                                        })
                                    );

                                    // renderPass.drawIndexedIndirect(indirectBuffer, indirectOffset)
                                    g_engine->setProperty(jsRenderPass, "drawIndexedIndirect",
                                        g_engine->newFunction("drawIndexedIndirect", [passResourceId](void* ctx, const std::vector<js::JSValueHandle>& args) {
                                            auto activeResource = resourceHandle<WGPURenderPassEncoder>(passResourceId);
                                            if (!activeResource) return g_engine->newUndefined();
                                            g_jsRenderPass = activeResource;
                                            if (args.size() < 2) return g_engine->newUndefined();

                                            WGPUBuffer indirectBuffer = (WGPUBuffer)resourceData(args[0]);
                                            uint64_t indirectOffset = (uint64_t)g_engine->toNumber(args[1]);

                                            if (g_jsRenderPass && indirectBuffer) {
                                                wgpuRenderPassEncoderDrawIndexedIndirect(g_jsRenderPass, indirectBuffer, indirectOffset);
                                                if (g_verboseLogging) std::cout << "[WebGPU] DrawIndexedIndirect at offset " << indirectOffset << std::endl;
                                            }

                                            return g_engine->newUndefined();
                                        })
                                    );

                                    // renderPass.setViewport(x, y, width, height, minDepth, maxDepth)
                                    g_engine->setProperty(jsRenderPass, "setViewport",
                                        g_engine->newFunction("setViewport", [passResourceId](void* ctx, const std::vector<js::JSValueHandle>& args) {
                                            auto activeResource = resourceHandle<WGPURenderPassEncoder>(passResourceId);
                                            if (!activeResource) return g_engine->newUndefined();
                                            g_jsRenderPass = activeResource;
                                            if (args.size() < 6) return g_engine->newUndefined();

                                            float x = (float)g_engine->toNumber(args[0]);
                                            float y = (float)g_engine->toNumber(args[1]);
                                            float width = (float)g_engine->toNumber(args[2]);
                                            float height = (float)g_engine->toNumber(args[3]);
                                            float minDepth = (float)g_engine->toNumber(args[4]);
                                            float maxDepth = (float)g_engine->toNumber(args[5]);

                                            if (g_jsRenderPass) {
                                                wgpuRenderPassEncoderSetViewport(g_jsRenderPass, x, y, width, height, minDepth, maxDepth);
                                                if (g_verboseLogging) std::cout << "[WebGPU] SetViewport: " << x << "," << y << " " << width << "x" << height << std::endl;
                                            }

                                            return g_engine->newUndefined();
                                        })
                                    );

                                    // renderPass.setScissorRect(x, y, width, height)
                                    g_engine->setProperty(jsRenderPass, "setScissorRect",
                                        g_engine->newFunction("setScissorRect", [passResourceId](void* ctx, const std::vector<js::JSValueHandle>& args) {
                                            auto activeResource = resourceHandle<WGPURenderPassEncoder>(passResourceId);
                                            if (!activeResource) return g_engine->newUndefined();
                                            g_jsRenderPass = activeResource;
                                            if (args.size() < 4) return g_engine->newUndefined();

                                            uint32_t x = (uint32_t)g_engine->toNumber(args[0]);
                                            uint32_t y = (uint32_t)g_engine->toNumber(args[1]);
                                            uint32_t width = (uint32_t)g_engine->toNumber(args[2]);
                                            uint32_t height = (uint32_t)g_engine->toNumber(args[3]);

                                            if (g_jsRenderPass) {
                                                wgpuRenderPassEncoderSetScissorRect(g_jsRenderPass, x, y, width, height);
                                                if (g_verboseLogging) std::cout << "[WebGPU] SetScissorRect: " << x << "," << y << " " << width << "x" << height << std::endl;
                                            }

                                            return g_engine->newUndefined();
                                        })
                                    );

                                    // renderPass.setBlendConstant(color)
                                    g_engine->setProperty(jsRenderPass, "setBlendConstant",
                                        g_engine->newFunction("setBlendConstant", [passResourceId](void* ctx, const std::vector<js::JSValueHandle>& args) {
                                            auto activeResource = resourceHandle<WGPURenderPassEncoder>(passResourceId);
                                            if (!activeResource) return g_engine->newUndefined();
                                            g_jsRenderPass = activeResource;
                                            if (args.empty()) return g_engine->newUndefined();

                                            auto color = args[0];
                                            WGPUColor blendColor = {};
                                            if (g_engine->isArray(color)) {
                                                blendColor.r = g_engine->toNumber(g_engine->getPropertyIndex(color, 0));
                                                blendColor.g = g_engine->toNumber(g_engine->getPropertyIndex(color, 1));
                                                blendColor.b = g_engine->toNumber(g_engine->getPropertyIndex(color, 2));
                                                blendColor.a = g_engine->toNumber(g_engine->getPropertyIndex(color, 3));
                                            } else {
                                                blendColor.r = g_engine->toNumber(g_engine->getProperty(color, "r"));
                                                blendColor.g = g_engine->toNumber(g_engine->getProperty(color, "g"));
                                                blendColor.b = g_engine->toNumber(g_engine->getProperty(color, "b"));
                                                blendColor.a = g_engine->toNumber(g_engine->getProperty(color, "a"));
                                            }

                                            if (g_jsRenderPass) {
                                                wgpuRenderPassEncoderSetBlendConstant(g_jsRenderPass, &blendColor);
                                            }

                                            return g_engine->newUndefined();
                                        })
                                    );

                                    // renderPass.setStencilReference(reference)
                                    g_engine->setProperty(jsRenderPass, "setStencilReference",
                                        g_engine->newFunction("setStencilReference", [passResourceId](void* ctx, const std::vector<js::JSValueHandle>& args) {
                                            auto activeResource = resourceHandle<WGPURenderPassEncoder>(passResourceId);
                                            if (!activeResource) return g_engine->newUndefined();
                                            g_jsRenderPass = activeResource;
                                            if (args.empty()) return g_engine->newUndefined();

                                            uint32_t reference = (uint32_t)g_engine->toNumber(args[0]);
                                            if (g_jsRenderPass) {
                                                wgpuRenderPassEncoderSetStencilReference(g_jsRenderPass, reference);
                                            }

                                            return g_engine->newUndefined();
                                        })
                                    );

                                    // renderPass.executeBundles(bundles)
                                    // Used by Three.js for mipmap generation
                                    WGPURenderPassEncoder capturedRenderPassForBundles = renderPass;
                                    g_engine->setProperty(jsRenderPass, "executeBundles",
                                        g_engine->newFunction("executeBundles", [capturedRenderPassForBundles, passResourceId](void* ctx, const std::vector<js::JSValueHandle>& args) {
                                            auto activeResource = resourceHandle<WGPURenderPassEncoder>(passResourceId);
                                            if (!activeResource) return g_engine->newUndefined();
                                            g_jsRenderPass = activeResource;
                                            if (args.empty() || !capturedRenderPassForBundles) return g_engine->newUndefined();

                                            auto bundlesArray = args[0];
                                            auto lengthProp = g_engine->getProperty(bundlesArray, "length");
                                            int bundleCount = g_engine->isUndefined(lengthProp) ? 0 : (int)g_engine->toNumber(lengthProp);

                                            std::vector<WGPURenderBundle> bundles;
                                            bundles.reserve(bundleCount);
                                            for (int i = 0; i < bundleCount; i++) {
                                                auto bundleHandle = g_engine->getPropertyIndex(bundlesArray, i);
                                                WGPURenderBundle bundle = (WGPURenderBundle)resourceData(bundleHandle);
                                                if (bundle) bundles.push_back(bundle);
                                            }

                                            if (!bundles.empty()) {
                                                wgpuRenderPassEncoderExecuteBundles(capturedRenderPassForBundles, bundles.size(), bundles.data());
                                                if (g_verboseLogging) std::cout << "[WebGPU] Executed " << bundles.size() << " render bundles" << std::endl;
                                            }

                                            return g_engine->newUndefined();
                                        })
                                    );

                                    // renderPass.end() - capture encoder and render pass for cleanup
                                    WGPUCommandEncoder capturedEncoderForEnd = encoderToUse;
                                    WGPURenderPassEncoder capturedRenderPass = renderPass;
                                    g_engine->setProperty(jsRenderPass, "end",
                                        g_engine->newFunction("end", [capturedEncoderForEnd, capturedRenderPass, passResourceId](void* ctx, const std::vector<js::JSValueHandle>& args) {
                                            auto activeResource = resourceHandle<WGPURenderPassEncoder>(passResourceId);
                                            if (!activeResource) return g_engine->newUndefined();
                                            g_jsRenderPass = activeResource;
                                            if (capturedRenderPass) {
                                                wgpuRenderPassEncoderEnd(capturedRenderPass);
                                                releaseReference(capturedRenderPass, ResourceKind::RenderPass);

                                                // Remove from per-encoder map
                                                g_encoderRenderPassMap.erase(capturedEncoderForEnd);

                                                // Clear global if it matches
                                                if (g_jsRenderPass == capturedRenderPass) {
                                                    g_jsRenderPass = nullptr;
                                                }

                                                // Mark surface render pass as ended
                                                if (g_surfaceRenderEncoder == capturedEncoderForEnd) {
                                                    g_surfaceRenderPassEnded = true;
                                                }
                                                if (g_verboseLogging) std::cout << "[WebGPU] Render pass ended" << std::endl;
                                            }
                                            return g_engine->newUndefined();
                                        })
                                    );

                                    // Resume frame tracking
                                    g_engine->resumeFrameTracking();

                                    return jsRenderPass;
                                })
                            );

                            // encoder.beginComputePass(descriptor?)
                            g_engine->setProperty(jsEncoder, "beginComputePass",
                                g_engine->newFunction("beginComputePass", [encoderResourceId](void* ctx, const std::vector<js::JSValueHandle>& args) {
                                            auto activeResource = resourceHandle<WGPUCommandEncoder>(encoderResourceId);
                                            if (!activeResource) return g_engine->newUndefined();
                                            g_jsCommandEncoder = activeResource;
                                    if (!g_jsCommandEncoder) {
                                        g_engine->throwException("No command encoder");
                                        return g_engine->newUndefined();
                                    }

                                    WGPUComputePassDescriptor computePassDesc = {};
                                    g_jsComputePass = wgpuCommandEncoderBeginComputePass(g_jsCommandEncoder, &computePassDesc);

                                    auto jsComputePass = g_engine->newObject();
                                    const auto passResourceId = manageResource(jsComputePass, g_jsComputePass, ResourceKind::ComputePass);
                                    g_encoderComputePassMap[g_jsCommandEncoder] = g_jsComputePass;
                                    const auto computeEncoder = g_jsCommandEncoder;

                                    // computePass.setPipeline(pipeline)
                                    g_engine->setProperty(jsComputePass, "setPipeline",
                                        g_engine->newFunction("setPipeline", [passResourceId](void* ctx, const std::vector<js::JSValueHandle>& args) {
                                            auto activeResource = resourceHandle<WGPUComputePassEncoder>(passResourceId);
                                            if (!activeResource) return g_engine->newUndefined();
                                            g_jsComputePass = activeResource;
                                            if (args.empty()) return g_engine->newUndefined();
                                            WGPUComputePipeline pipeline = (WGPUComputePipeline)resourceData(args[0]);
                                            if (g_jsComputePass && pipeline) {
                                                wgpuComputePassEncoderSetPipeline(g_jsComputePass, pipeline);
                                            }
                                            return g_engine->newUndefined();
                                        })
                                    );

                                    // computePass.setBindGroup(index, bindGroup, dynamicOffsets?)
                                    g_engine->setProperty(jsComputePass, "setBindGroup",
                                        g_engine->newFunction("setBindGroup", [passResourceId](void* ctx, const std::vector<js::JSValueHandle>& args) {
                                            auto activeResource = resourceHandle<WGPUComputePassEncoder>(passResourceId);
                                            if (!activeResource) return g_engine->newUndefined();
                                            g_jsComputePass = activeResource;
                                            if (args.size() < 2) {
                                                g_engine->throwException("setBindGroup requires an index and nullable bind group");
                                                return g_engine->newUndefined();
                                            }
                                            uint64_t index;
                                            if (!copyUnsigned(args[0], 0, false, index)) return g_engine->newUndefined();
                                            WGPUBindGroup bindGroup = g_engine->isNull(args[1]) || g_engine->isUndefined(args[1]) ?
                                                nullptr : static_cast<WGPUBindGroup>(resourceData(args[1]));
                                            std::vector<uint32_t> offsets;
                                            if (!dynamicOffsets(args, offsets)) return g_engine->newUndefined();
                                            if (g_jsComputePass) {
                                                wgpuComputePassEncoderSetBindGroup(g_jsComputePass, static_cast<uint32_t>(index),
                                                    bindGroup, offsets.size(), offsets.data());
                                            }
                                            return g_engine->newUndefined();
                                        })
                                    );

                                    // computePass.dispatchWorkgroups(countX, countY?, countZ?)
                                    g_engine->setProperty(jsComputePass, "dispatchWorkgroups",
                                        g_engine->newFunction("dispatchWorkgroups", [passResourceId](void* ctx, const std::vector<js::JSValueHandle>& args) {
                                            auto activeResource = resourceHandle<WGPUComputePassEncoder>(passResourceId);
                                            if (!activeResource) return g_engine->newUndefined();
                                            g_jsComputePass = activeResource;
                                            if (args.empty()) return g_engine->newUndefined();
                                            uint32_t countX = (uint32_t)g_engine->toNumber(args[0]);
                                            uint32_t countY = args.size() > 1 ? (uint32_t)g_engine->toNumber(args[1]) : 1;
                                            uint32_t countZ = args.size() > 2 ? (uint32_t)g_engine->toNumber(args[2]) : 1;
                                            if (g_jsComputePass) {
                                                wgpuComputePassEncoderDispatchWorkgroups(g_jsComputePass, countX, countY, countZ);
                                            }
                                            return g_engine->newUndefined();
                                        })
                                    );

                                    g_engine->setProperty(jsComputePass, "dispatchWorkgroupsIndirect",
                                        g_engine->newFunction("dispatchWorkgroupsIndirect", [passResourceId](void*, const std::vector<js::JSValueHandle>& args) {
                                            auto activeResource = resourceHandle<WGPUComputePassEncoder>(passResourceId);
                                            if (!activeResource) return g_engine->newUndefined();
                                            g_jsComputePass = activeResource;
                                            if (args.size() < 2 || g_engine->isUndefined(args[1]) || !g_jsComputePass) {
                                                g_engine->throwException("dispatchWorkgroupsIndirect requires an active pass, buffer, and byte offset");
                                                return g_engine->newUndefined();
                                            }
                                            if (!g_engine->isObject(args[0])) {
                                                g_engine->throwException("dispatchWorkgroupsIndirect requires a GPUBuffer");
                                                return g_engine->newUndefined();
                                            }
                                            auto buffer = static_cast<WGPUBuffer>(resourceData(args[0]));
                                            uint64_t offset;
                                            if (!copyUnsigned(args[1], 0, true, offset)) return g_engine->newUndefined();
                                            if (!buffer) {
                                                g_engine->throwException("dispatchWorkgroupsIndirect requires a GPUBuffer");
                                                return g_engine->newUndefined();
                                            }
                                            wgpuComputePassEncoderDispatchWorkgroupsIndirect(g_jsComputePass, buffer, offset);
                                            return g_engine->newUndefined();
                                        }));

                                    // computePass.end()
                                    g_engine->setProperty(jsComputePass, "end",
                                        g_engine->newFunction("end", [computeEncoder, passResourceId](void* ctx, const std::vector<js::JSValueHandle>& args) {
                                            auto activeResource = resourceHandle<WGPUComputePassEncoder>(passResourceId);
                                            if (!activeResource) return g_engine->newUndefined();
                                            g_jsComputePass = activeResource;
                                            if (g_jsComputePass) {
                                                wgpuComputePassEncoderEnd(g_jsComputePass);
                                                releaseReference(g_jsComputePass, ResourceKind::ComputePass);
                                                g_encoderComputePassMap.erase(computeEncoder);
                                                g_jsComputePass = nullptr;
                                            }
                                            return g_engine->newUndefined();
                                        })
                                    );

                                    if (g_verboseLogging) std::cout << "[WebGPU] Compute pass started" << std::endl;
                                    return jsComputePass;
                                })
                            );

                            // encoder.copyBufferToBuffer(source, sourceOffset, destination, destinationOffset, size)
                            g_engine->setProperty(jsEncoder, "copyBufferToBuffer",
                                g_engine->newFunction("copyBufferToBuffer", [encoderResourceId](void* ctx, const std::vector<js::JSValueHandle>& args) {
                                            auto activeResource = resourceHandle<WGPUCommandEncoder>(encoderResourceId);
                                            if (!activeResource) return g_engine->newUndefined();
                                            g_jsCommandEncoder = activeResource;
                                    if (args.size() < 5 || !g_jsCommandEncoder) return g_engine->newUndefined();

                                    WGPUBuffer source = (WGPUBuffer)resourceData(args[0]);
                                    uint64_t sourceOffset = (uint64_t)g_engine->toNumber(args[1]);
                                    WGPUBuffer destination = (WGPUBuffer)resourceData(args[2]);
                                    uint64_t destOffset = (uint64_t)g_engine->toNumber(args[3]);
                                    uint64_t size = (uint64_t)g_engine->toNumber(args[4]);

                                    if (source && destination) {
                                        wgpuCommandEncoderCopyBufferToBuffer(g_jsCommandEncoder, source, sourceOffset, destination, destOffset, size);
                                    }
                                    return g_engine->newUndefined();
                                })
                            );

                            // Texture copies share WebIDL defaults and preserve unspecified native strides.
                            g_engine->setProperty(jsEncoder, "copyBufferToTexture",
                                g_engine->newFunction("copyBufferToTexture", [encoderResourceId](void*, const std::vector<js::JSValueHandle>& args) {
                                            auto activeResource = resourceHandle<WGPUCommandEncoder>(encoderResourceId);
                                            if (!activeResource) return g_engine->newUndefined();
                                            g_jsCommandEncoder = activeResource;
                                    if (args.size() < 3 || !g_jsCommandEncoder) return g_engine->newUndefined();
                                    WGPUImageCopyBuffer_Compat source = {};
                                    source.buffer = static_cast<WGPUBuffer>(resourceData(g_engine->getProperty(args[0], "buffer")));
                                    WGPUImageCopyTexture_Compat destination = {};
                                    WGPUExtent3D extent = {};
                                    if (!copyLayout(args[0], source.layout) || !copyTexture(args[1], destination) ||
                                        !copyExtent(args[2], extent)) return g_engine->newUndefined();
                                    if (source.buffer && destination.texture)
                                        wgpuCommandEncoderCopyBufferToTexture(g_jsCommandEncoder, &source, &destination, &extent);
                                    return g_engine->newUndefined();
                                }));
                            g_engine->setProperty(jsEncoder, "copyTextureToBuffer",
                                g_engine->newFunction("copyTextureToBuffer", [encoderResourceId](void*, const std::vector<js::JSValueHandle>& args) {
                                            auto activeResource = resourceHandle<WGPUCommandEncoder>(encoderResourceId);
                                            if (!activeResource) return g_engine->newUndefined();
                                            g_jsCommandEncoder = activeResource;
                                    if (args.size() < 3 || !g_jsCommandEncoder) return g_engine->newUndefined();
                                    WGPUImageCopyTexture_Compat source = {};
                                    WGPUImageCopyBuffer_Compat destination = {};
                                    destination.buffer = static_cast<WGPUBuffer>(resourceData(g_engine->getProperty(args[1], "buffer")));
                                    WGPUExtent3D extent = {};
                                    if (!copyTexture(args[0], source) || !copyLayout(args[1], destination.layout) ||
                                        !copyExtent(args[2], extent)) return g_engine->newUndefined();
                                    if (source.texture && destination.buffer)
                                        wgpuCommandEncoderCopyTextureToBuffer(g_jsCommandEncoder, &source, &destination, &extent);
                                    return g_engine->newUndefined();
                                }));
                            g_engine->setProperty(jsEncoder, "copyTextureToTexture",
                                g_engine->newFunction("copyTextureToTexture", [encoderResourceId](void*, const std::vector<js::JSValueHandle>& args) {
                                            auto activeResource = resourceHandle<WGPUCommandEncoder>(encoderResourceId);
                                            if (!activeResource) return g_engine->newUndefined();
                                            g_jsCommandEncoder = activeResource;
                                    if (args.size() < 3 || !g_jsCommandEncoder) return g_engine->newUndefined();
                                    WGPUImageCopyTexture_Compat source = {}, destination = {};
                                    WGPUExtent3D extent = {};
                                    if (!copyTexture(args[0], source) || !copyTexture(args[1], destination) ||
                                        !copyExtent(args[2], extent)) return g_engine->newUndefined();
                                    if (source.texture && destination.texture)
                                        wgpuCommandEncoderCopyTextureToTexture(g_jsCommandEncoder, &source, &destination, &extent);
                                    return g_engine->newUndefined();
                                }));

                            // encoder.clearBuffer(buffer, offset?, size?)
                            g_engine->setProperty(jsEncoder, "clearBuffer",
                                g_engine->newFunction("clearBuffer", [encoderResourceId](void* ctx, const std::vector<js::JSValueHandle>& args) {
                                            auto activeResource = resourceHandle<WGPUCommandEncoder>(encoderResourceId);
                                            if (!activeResource) return g_engine->newUndefined();
                                            g_jsCommandEncoder = activeResource;
                                    if (args.empty() || !g_jsCommandEncoder) return g_engine->newUndefined();

                                    WGPUBuffer buffer = (WGPUBuffer)resourceData(args[0]);
                                    uint64_t offset = args.size() > 1 ? (uint64_t)g_engine->toNumber(args[1]) : 0;
                                    uint64_t size = args.size() > 2 ? (uint64_t)g_engine->toNumber(args[2]) : WGPU_WHOLE_SIZE;

                                    if (buffer) {
                                        wgpuCommandEncoderClearBuffer(g_jsCommandEncoder, buffer, offset, size);
                                    }
                                    return g_engine->newUndefined();
                                })
                            );

                            // encoder.finish(descriptor?)
                            g_engine->setProperty(jsEncoder, "finish",
                                g_engine->newFunction("finish", [capturedEncoder, encoderResourceId](void* ctx, const std::vector<js::JSValueHandle>& args) {
                                            auto activeResource = resourceHandle<WGPUCommandEncoder>(encoderResourceId);
                                            if (!activeResource) return g_engine->newUndefined();
                                            g_jsCommandEncoder = activeResource;
                                    // Use captured encoder for this specific command encoder
                                    WGPUCommandEncoder encoderToFinish = capturedEncoder;

                                    // Auto-end any active render/compute passes for THIS encoder
                                    // Look up from per-encoder map, not global
                                    auto renderPassIt = g_encoderRenderPassMap.find(encoderToFinish);
                                    if (renderPassIt != g_encoderRenderPassMap.end() && renderPassIt->second) {
                                        WGPURenderPassEncoder renderPass = renderPassIt->second;
                                        if (g_verboseLogging) std::cout << "[WebGPU] Auto-ending render pass (pass=" << (void*)renderPass << ", encoder=" << (void*)encoderToFinish << ")" << std::endl;
                                        wgpuRenderPassEncoderEnd(renderPass);
                                        releaseReference(renderPass, ResourceKind::RenderPass);
                                        g_encoderRenderPassMap.erase(encoderToFinish);

                                        // Clear global if it matches
                                        if (g_jsRenderPass == renderPass) {
                                            g_jsRenderPass = nullptr;
                                        }

                                        // Mark surface render pass as ended
                                        if (g_surfaceRenderEncoder == encoderToFinish) {
                                            g_surfaceRenderPassEnded = true;
                                            if (g_verboseLogging) std::cout << "[WebGPU] Surface render pass auto-ended (encoder=" << (void*)encoderToFinish << ")" << std::endl;
                                        }
                                    }

                                    auto computePassIt = g_encoderComputePassMap.find(encoderToFinish);
                                    if (computePassIt != g_encoderComputePassMap.end() && computePassIt->second) {
                                        WGPUComputePassEncoder computePass = computePassIt->second;
                                        if (g_verboseLogging) std::cout << "[WebGPU] Auto-ending compute pass (pass=" << (void*)computePass << ", encoder=" << (void*)encoderToFinish << ")" << std::endl;
                                        wgpuComputePassEncoderEnd(computePass);
                                        releaseReference(computePass, ResourceKind::ComputePass);
                                        g_encoderComputePassMap.erase(encoderToFinish);

                                        // Clear global if it matches
                                        if (g_jsComputePass == computePass) {
                                            g_jsComputePass = nullptr;
                                        }
                                    }

                                    WGPUCommandBufferDescriptor cmdDesc = {};
                                    WGPUCommandBuffer cmdBuffer = nullptr;

                                    if (encoderToFinish) {
                                        cmdBuffer = wgpuCommandEncoderFinish(encoderToFinish, &cmdDesc);
                                        releaseReference(encoderToFinish, ResourceKind::CommandEncoder);

                                        // Clear global if it matches
                                        if (g_jsCommandEncoder == encoderToFinish) {
                                            g_jsCommandEncoder = nullptr;
                                        }

                                        if (g_verboseLogging) std::cout << "[WebGPU] Command encoder finished, buffer: " << cmdBuffer << std::endl;
                                    }

                                    auto jsCommandBuffer = g_engine->newObject();
                                    manageResource(jsCommandBuffer, cmdBuffer, ResourceKind::CommandBuffer);

                                    return jsCommandBuffer;
                                })
                            );

                            // Resume frame tracking now that encoder wrapper is created
                            g_engine->resumeFrameTracking();

                            return jsEncoder;
                        })
                    );

                    // device.createTexture(descriptor)
                    g_engine->setProperty(device, "createTexture",
                        g_engine->newFunction("createTexture", [](void* ctx, const std::vector<js::JSValueHandle>& args) {
                            if (args.empty()) {
                                g_engine->throwException("createTexture requires a descriptor");
                                return g_engine->newUndefined();
                            }

                            auto descriptor = args[0];

                            // Parse size - can be [width, height, depth] array or {width, height, depthOrArrayLayers} object
                            auto sizeVal = g_engine->getProperty(descriptor, "size");
                            uint32_t width = 1, height = 1, depthOrArrayLayers = 1;

                            // Check if size is an array
                            auto lengthProp = g_engine->getProperty(sizeVal, "length");
                            if (!g_engine->isUndefined(lengthProp)) {
                                // Array format: [width, height?, depth?]
                                int len = (int)g_engine->toNumber(lengthProp);
                                if (len >= 1) width = (uint32_t)g_engine->toNumber(g_engine->getPropertyIndex(sizeVal, 0));
                                if (len >= 2) height = (uint32_t)g_engine->toNumber(g_engine->getPropertyIndex(sizeVal, 1));
                                if (len >= 3) depthOrArrayLayers = (uint32_t)g_engine->toNumber(g_engine->getPropertyIndex(sizeVal, 2));
                            } else {
                                // Object format: {width, height, depthOrArrayLayers}
                                auto w = g_engine->getProperty(sizeVal, "width");
                                auto h = g_engine->getProperty(sizeVal, "height");
                                auto d = g_engine->getProperty(sizeVal, "depthOrArrayLayers");
                                if (!g_engine->isUndefined(w)) width = (uint32_t)g_engine->toNumber(w);
                                if (!g_engine->isUndefined(h)) height = (uint32_t)g_engine->toNumber(h);
                                if (!g_engine->isUndefined(d)) depthOrArrayLayers = (uint32_t)g_engine->toNumber(d);
                            }

                            // Parse format
                            std::string formatStr = g_engine->toString(g_engine->getProperty(descriptor, "format"));
                            WGPUTextureFormat format = stringToFormat(formatStr);

                            // Parse usage
                            double usageVal = g_engine->toNumber(g_engine->getProperty(descriptor, "usage"));
                            WGPUTextureUsage usage = (WGPUTextureUsage)(uint32_t)usageVal;

                            // Fix format/usage incompatibility:
                            // BGRA8UnormSrgb doesn't support StorageBinding, convert to BGRA8Unorm or RGBA8Unorm
                            if (format == WGPUTextureFormat_BGRA8UnormSrgb && (usage & WGPUTextureUsage_StorageBinding)) {
                                std::cout << "[WebGPU] Warning: BGRA8UnormSrgb doesn't support StorageBinding, using RGBA8Unorm instead" << std::endl;
                                format = WGPUTextureFormat_RGBA8Unorm;
                                formatStr = "rgba8unorm";
                            }
                            // Also handle BGRA8Unorm which may not support storage on all platforms
                            if (format == WGPUTextureFormat_BGRA8Unorm && (usage & WGPUTextureUsage_StorageBinding)) {
                                std::cout << "[WebGPU] Warning: BGRA8Unorm may not support StorageBinding, using RGBA8Unorm instead" << std::endl;
                                format = WGPUTextureFormat_RGBA8Unorm;
                                formatStr = "rgba8unorm";
                            }

                            // Parse optional properties
                            std::string dimensionStr = g_engine->toString(g_engine->getProperty(descriptor, "dimension"));
                            WGPUTextureDimension dimension = dimensionStr.empty() ? WGPUTextureDimension_2D : stringToTextureDimension(dimensionStr);

                            auto mipLevelCountVal = g_engine->getProperty(descriptor, "mipLevelCount");
                            uint32_t mipLevelCount = g_engine->isUndefined(mipLevelCountVal) ? 1 : (uint32_t)g_engine->toNumber(mipLevelCountVal);

                            auto sampleCountVal = g_engine->getProperty(descriptor, "sampleCount");
                            uint32_t sampleCount = g_engine->isUndefined(sampleCountVal) ? 1 : (uint32_t)g_engine->toNumber(sampleCountVal);

                            // Create texture descriptor
                            WGPUTextureDescriptor texDesc = {};
                            texDesc.size.width = width;
                            texDesc.size.height = height;
                            texDesc.size.depthOrArrayLayers = depthOrArrayLayers;
                            texDesc.format = format;
                            texDesc.usage = usage;
                            texDesc.dimension = dimension;
                            texDesc.mipLevelCount = mipLevelCount;
                            texDesc.sampleCount = sampleCount;

                            WGPUTexture texture = wgpuDeviceCreateTexture(g_device, &texDesc);

                            if (!texture) {
                                g_engine->throwException("Failed to create texture");
                                return g_engine->newUndefined();
                            }

                            // Create JS wrapper
                            auto jsTexture = g_engine->newObject();
                            // The registry and its single owned API reference are registered below.

                            // Store texture properties
                            g_engine->setProperty(jsTexture, "width", g_engine->newNumber(width));
                            g_engine->setProperty(jsTexture, "height", g_engine->newNumber(height));
                            g_engine->setProperty(jsTexture, "depthOrArrayLayers", g_engine->newNumber(depthOrArrayLayers));
                            g_engine->setProperty(jsTexture, "format", g_engine->newString(formatStr.c_str()));
                            g_engine->setProperty(jsTexture, "mipLevelCount", g_engine->newNumber(mipLevelCount));
                            g_engine->setProperty(jsTexture, "sampleCount", g_engine->newNumber(sampleCount));

                            // Register texture for lookup by createView
                            uint64_t textureId = g_nextTextureId++;
                            g_textureRegistry[textureId] = {texture, format, width, height, depthOrArrayLayers, mipLevelCount, dimension};
                            manageResource(jsTexture, texture, ResourceKind::Texture, textureId);

                            // Store texture ID for lookup
                            g_engine->setProperty(jsTexture, "_textureId", g_engine->newNumber((double)textureId));

                            // texture.createView(descriptor?) - Store texture ID for lookup
                            // We store the textureId to look up the texture later since callbacks don't have 'this'
                            g_engine->setProperty(jsTexture, "_createViewTextureId", g_engine->newNumber((double)textureId));

                            g_engine->setProperty(jsTexture, "createView",
                                g_engine->newFunction("createView", [textureId](void* ctx, const std::vector<js::JSValueHandle>& args) {
                                    // Look up texture from registry using captured textureId
                                    auto it = g_textureRegistry.find(textureId);
                                    if (it == g_textureRegistry.end()) {
                                        std::cerr << "[WebGPU] createView: Texture " << textureId << " not found in registry" << std::endl;
                                        return g_engine->newUndefined();
                                    }

                                    WGPUTexture texture = resourceHandle<WGPUTexture>(it->second.resourceId);
                                    if (!texture) {
                                        std::cerr << "[WebGPU] createView: Texture " << textureId << " is null" << std::endl;
                                        return g_engine->newUndefined();
                                    }

                                    WGPUTextureViewDescriptor viewDesc = {};
                                    // Default values - use all mips and layers from the texture
                                    viewDesc.format = it->second.format;
                                    viewDesc.mipLevelCount = it->second.mipLevelCount > 0 ? it->second.mipLevelCount : 1;
                                    viewDesc.baseMipLevel = 0;
                                    viewDesc.baseArrayLayer = 0;
                                    viewDesc.aspect = WGPUTextureAspect_All;

                                    // Default dimension and arrayLayerCount based on texture dimension
                                    if (it->second.dimension == WGPUTextureDimension_3D) {
                                        // 3D textures: view as 3D, arrayLayerCount must be 1
                                        viewDesc.dimension = WGPUTextureViewDimension_3D;
                                        viewDesc.arrayLayerCount = 1;
                                    } else if (it->second.dimension == WGPUTextureDimension_1D) {
                                        // 1D textures
                                        viewDesc.dimension = WGPUTextureViewDimension_1D;
                                        viewDesc.arrayLayerCount = 1;
                                    } else {
                                        // 2D textures: use layers for 2D-array, 1 for regular 2D
                                        viewDesc.arrayLayerCount = it->second.depthOrArrayLayers > 0 ? it->second.depthOrArrayLayers : 1;
                                        viewDesc.dimension = it->second.depthOrArrayLayers > 1 ? WGPUTextureViewDimension_2DArray : WGPUTextureViewDimension_2D;
                                    }

                                    // Parse view descriptor if provided
                                    if (!args.empty() && !g_engine->isUndefined(args[0])) {
                                        auto descriptor = args[0];

                                        // format (optional, defaults to texture format)
                                        auto formatProp = g_engine->getProperty(descriptor, "format");
                                        if (!g_engine->isUndefined(formatProp)) {
                                            viewDesc.format = stringToFormat(g_engine->toString(formatProp));
                                        } else {
                                            viewDesc.format = it->second.format;
                                        }

                                        // dimension (optional)
                                        auto dimensionProp = g_engine->getProperty(descriptor, "dimension");
                                        if (!g_engine->isUndefined(dimensionProp)) {
                                            std::string dimStr = g_engine->toString(dimensionProp);
                                            if (dimStr == "1d") viewDesc.dimension = WGPUTextureViewDimension_1D;
                                            else if (dimStr == "2d") viewDesc.dimension = WGPUTextureViewDimension_2D;
                                            else if (dimStr == "2d-array") viewDesc.dimension = WGPUTextureViewDimension_2DArray;
                                            else if (dimStr == "cube") viewDesc.dimension = WGPUTextureViewDimension_Cube;
                                            else if (dimStr == "cube-array") viewDesc.dimension = WGPUTextureViewDimension_CubeArray;
                                            else if (dimStr == "3d") viewDesc.dimension = WGPUTextureViewDimension_3D;
                                        }

                                        // aspect (optional)
                                        auto aspectProp = g_engine->getProperty(descriptor, "aspect");
                                        if (!g_engine->isUndefined(aspectProp)) {
                                            std::string aspectStr = g_engine->toString(aspectProp);
                                            if (aspectStr == "all") viewDesc.aspect = WGPUTextureAspect_All;
                                            else if (aspectStr == "stencil-only") viewDesc.aspect = WGPUTextureAspect_StencilOnly;
                                            else if (aspectStr == "depth-only") viewDesc.aspect = WGPUTextureAspect_DepthOnly;
                                        }

                                        // baseMipLevel (optional)
                                        auto baseMipProp = g_engine->getProperty(descriptor, "baseMipLevel");
                                        if (!g_engine->isUndefined(baseMipProp)) {
                                            viewDesc.baseMipLevel = (uint32_t)g_engine->toNumber(baseMipProp);
                                        }

                                        // mipLevelCount (optional)
                                        auto mipCountProp = g_engine->getProperty(descriptor, "mipLevelCount");
                                        if (!g_engine->isUndefined(mipCountProp)) {
                                            viewDesc.mipLevelCount = (uint32_t)g_engine->toNumber(mipCountProp);
                                        }

                                        // baseArrayLayer (optional)
                                        auto baseLayerProp = g_engine->getProperty(descriptor, "baseArrayLayer");
                                        if (!g_engine->isUndefined(baseLayerProp)) {
                                            viewDesc.baseArrayLayer = (uint32_t)g_engine->toNumber(baseLayerProp);
                                        }

                                        // arrayLayerCount (optional)
                                        auto layerCountProp = g_engine->getProperty(descriptor, "arrayLayerCount");
                                        if (!g_engine->isUndefined(layerCountProp)) {
                                            uint32_t requested = (uint32_t)g_engine->toNumber(layerCountProp);
                                            uint32_t maxLayers = it->second.depthOrArrayLayers > 0 ? it->second.depthOrArrayLayers : 1;
                                            // Clamp to actual texture layer count
                                            viewDesc.arrayLayerCount = std::min(requested, maxLayers - viewDesc.baseArrayLayer);
                                        }
                                    }
                                    // else: defaults are already set above

                                    // Final validation: Fix arrayLayerCount based on view dimension
                                    if (viewDesc.dimension == WGPUTextureViewDimension_3D ||
                                        viewDesc.dimension == WGPUTextureViewDimension_1D) {
                                        // 3D/1D textures have no array layers
                                        viewDesc.arrayLayerCount = 1;
                                    } else if (viewDesc.dimension == WGPUTextureViewDimension_Cube) {
                                        // Cube requires exactly 6 layers (the 6 faces)
                                        viewDesc.arrayLayerCount = 6;
                                    } else if (viewDesc.dimension == WGPUTextureViewDimension_CubeArray) {
                                        // CubeArray must have multiple of 6 layers
                                        uint32_t maxLayers = it->second.depthOrArrayLayers > 0 ? it->second.depthOrArrayLayers : 6;
                                        viewDesc.arrayLayerCount = std::min(viewDesc.arrayLayerCount, maxLayers);
                                        // Round down to nearest multiple of 6
                                        viewDesc.arrayLayerCount = (viewDesc.arrayLayerCount / 6) * 6;
                                        if (viewDesc.arrayLayerCount < 6) viewDesc.arrayLayerCount = 6;
                                    }

                                    WGPUTextureView view = wgpuTextureCreateView(texture, &viewDesc);

                                    if (!view) {
                                        std::cerr << "[WebGPU] createView: Failed to create texture view" << std::endl;
                                        return g_engine->newUndefined();
                                    }

                                    auto jsView = g_engine->newObject();
                                    manageTextureView(jsView, view, texture);
                                    g_engine->setProperty(jsView, "_type", g_engine->newString("textureView"));


                                    return jsView;
                                })
                            );

                            // texture.destroy()
                            g_engine->setProperty(jsTexture, "destroy",
                                g_engine->newFunction("destroy", [textureId](void* ctx, const std::vector<js::JSValueHandle>& args) {
                                    const auto it = g_textureRegistry.find(textureId);
                                    if (it != g_textureRegistry.end()) releaseResource(it->second.resourceId, true);
                                    return g_engine->newUndefined();
                                })
                            );

                            if (g_verboseLogging) std::cout << "[WebGPU] Created texture " << width << "x" << height << " format=" << formatStr << " (id=" << textureId << ")" << std::endl;
                            return jsTexture;
                        })
                    );

                    // device.createSampler(descriptor?)
                    g_engine->setProperty(device, "createSampler",
                        g_engine->newFunction("createSampler", [](void* ctx, const std::vector<js::JSValueHandle>& args) {
                            WGPUSamplerDescriptor samplerDesc = {};

                            // Default values
                            samplerDesc.addressModeU = WGPUAddressMode_ClampToEdge;
                            samplerDesc.addressModeV = WGPUAddressMode_ClampToEdge;
                            samplerDesc.addressModeW = WGPUAddressMode_ClampToEdge;
                            samplerDesc.magFilter = WGPUFilterMode_Nearest;
                            samplerDesc.minFilter = WGPUFilterMode_Nearest;
                            samplerDesc.mipmapFilter = WGPUMipmapFilterMode_Nearest;
                            samplerDesc.lodMinClamp = 0.0f;
                            samplerDesc.lodMaxClamp = 32.0f;
                            samplerDesc.maxAnisotropy = 1;

                            if (!args.empty()) {
                                auto descriptor = args[0];

                                auto addressModeU = g_engine->getProperty(descriptor, "addressModeU");
                                if (!g_engine->isUndefined(addressModeU)) {
                                    samplerDesc.addressModeU = stringToAddressMode(g_engine->toString(addressModeU));
                                }

                                auto addressModeV = g_engine->getProperty(descriptor, "addressModeV");
                                if (!g_engine->isUndefined(addressModeV)) {
                                    samplerDesc.addressModeV = stringToAddressMode(g_engine->toString(addressModeV));
                                }

                                auto addressModeW = g_engine->getProperty(descriptor, "addressModeW");
                                if (!g_engine->isUndefined(addressModeW)) {
                                    samplerDesc.addressModeW = stringToAddressMode(g_engine->toString(addressModeW));
                                }

                                auto magFilter = g_engine->getProperty(descriptor, "magFilter");
                                if (!g_engine->isUndefined(magFilter)) {
                                    samplerDesc.magFilter = stringToFilterMode(g_engine->toString(magFilter));
                                }

                                auto minFilter = g_engine->getProperty(descriptor, "minFilter");
                                if (!g_engine->isUndefined(minFilter)) {
                                    samplerDesc.minFilter = stringToFilterMode(g_engine->toString(minFilter));
                                }

                                auto mipmapFilter = g_engine->getProperty(descriptor, "mipmapFilter");
                                if (!g_engine->isUndefined(mipmapFilter)) {
                                    samplerDesc.mipmapFilter = stringToMipmapFilterMode(g_engine->toString(mipmapFilter));
                                }

                                auto lodMinClamp = g_engine->getProperty(descriptor, "lodMinClamp");
                                if (!g_engine->isUndefined(lodMinClamp)) {
                                    samplerDesc.lodMinClamp = (float)g_engine->toNumber(lodMinClamp);
                                }

                                auto lodMaxClamp = g_engine->getProperty(descriptor, "lodMaxClamp");
                                if (!g_engine->isUndefined(lodMaxClamp)) {
                                    samplerDesc.lodMaxClamp = (float)g_engine->toNumber(lodMaxClamp);
                                }

                                auto compare = g_engine->getProperty(descriptor, "compare");
                                if (!g_engine->isUndefined(compare)) {
                                    samplerDesc.compare = stringToCompareFunction(g_engine->toString(compare));
                                }

                                auto maxAnisotropy = g_engine->getProperty(descriptor, "maxAnisotropy");
                                if (!g_engine->isUndefined(maxAnisotropy)) {
                                    samplerDesc.maxAnisotropy = (uint16_t)g_engine->toNumber(maxAnisotropy);
                                }
                            }

                            WGPUSampler sampler = wgpuDeviceCreateSampler(g_device, &samplerDesc);

                            auto jsSampler = g_engine->newObject();
                            manageResource(jsSampler, sampler, ResourceKind::Sampler);
                            g_engine->setProperty(jsSampler, "_type", g_engine->newString("sampler"));

                            if (g_verboseLogging) std::cout << "[WebGPU] Created sampler" << std::endl;
                            return jsSampler;
                        })
                    );

                    // device.createBindGroupLayout(descriptor)
                    g_engine->setProperty(device, "createBindGroupLayout",
                        g_engine->newFunction("createBindGroupLayout", [](void* ctx, const std::vector<js::JSValueHandle>& args) {
                            if (args.empty()) {
                                g_engine->throwException("createBindGroupLayout requires a descriptor");
                                return g_engine->newUndefined();
                            }

                            auto descriptor = args[0];
                            auto entries = g_engine->getProperty(descriptor, "entries");
                            if (!sequenceArray(entries, entries)) return g_engine->newUndefined();
                            auto lengthProp = g_engine->getProperty(entries, "length");
                            int entryCount = g_engine->isUndefined(lengthProp) ? 0 : (int)g_engine->toNumber(lengthProp);

                            std::vector<WGPUBindGroupLayoutEntry> layoutEntries;
                            layoutEntries.reserve(entryCount);
                            std::vector<WGPUExternalTextureBindingLayout> externalTextureLayouts;
                            externalTextureLayouts.reserve(entryCount);

                            for (int i = 0; i < entryCount; i++) {
                                auto entry = g_engine->getPropertyIndex(entries, i);

                                WGPUBindGroupLayoutEntry layoutEntry = {};
                                layoutEntry.binding = (uint32_t)g_engine->toNumber(g_engine->getProperty(entry, "binding"));
                                layoutEntry.visibility = (WGPUShaderStage)(uint32_t)g_engine->toNumber(g_engine->getProperty(entry, "visibility"));

                                // Check for buffer binding
                                auto buffer = g_engine->getProperty(entry, "buffer");
                                if (!g_engine->isUndefined(buffer)) {
                                    auto typeProp = g_engine->getProperty(buffer, "type");
                                    std::string typeStr = g_engine->isUndefined(typeProp) ? "" : g_engine->toString(typeProp);
                                    if (typeStr == "uniform" || typeStr == "") {
                                        // Default to uniform if no type specified (Three.js uses empty {})
                                        layoutEntry.buffer.type = WGPUBufferBindingType_Uniform;
                                    } else if (typeStr == "storage") {
                                        layoutEntry.buffer.type = WGPUBufferBindingType_Storage;
                                    } else if (typeStr == "read-only-storage") {
                                        layoutEntry.buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
                                    } else {
                                        // Default to uniform for unknown types
                                        layoutEntry.buffer.type = WGPUBufferBindingType_Uniform;
                                    }
                                    auto dynamic = g_engine->getProperty(buffer, "hasDynamicOffset");
                                    layoutEntry.buffer.hasDynamicOffset = !g_engine->isUndefined(dynamic) && g_engine->toBoolean(dynamic);
                                    uint64_t minimumSize;
                                    if (!copyUnsigned(g_engine->getProperty(buffer, "minBindingSize"), 0, true, minimumSize))
                                        return g_engine->newUndefined();
                                    layoutEntry.buffer.minBindingSize = minimumSize;
                                    std::cout << "[WebGPU] BindGroupLayout buffer binding "
                                              << layoutEntry.binding << ": type='" << typeStr
                                              << "', enum=" << static_cast<int>(layoutEntry.buffer.type)
                                              << std::endl;
                                }

                                // Check for sampler binding
                                auto sampler = g_engine->getProperty(entry, "sampler");
                                if (!g_engine->isUndefined(sampler)) {
                                    std::string typeStr = g_engine->toString(g_engine->getProperty(sampler, "type"));
                                    if (typeStr == "filtering") {
                                        layoutEntry.sampler.type = WGPUSamplerBindingType_Filtering;
                                    } else if (typeStr == "non-filtering") {
                                        layoutEntry.sampler.type = WGPUSamplerBindingType_NonFiltering;
                                    } else if (typeStr == "comparison") {
                                        layoutEntry.sampler.type = WGPUSamplerBindingType_Comparison;
                                    } else {
                                        // Default to filtering
                                        layoutEntry.sampler.type = WGPUSamplerBindingType_Filtering;
                                    }
                                }

                                // Check for texture binding
                                auto texture = g_engine->getProperty(entry, "texture");
                                if (!g_engine->isUndefined(texture)) {
                                    auto sampleTypeProp = g_engine->getProperty(texture, "sampleType");
                                    std::string sampleType = g_engine->isUndefined(sampleTypeProp) ? "" : g_engine->toString(sampleTypeProp);
                                    if (sampleType == "float" || sampleType == "") {
                                        // Default to float if no type specified (Three.js uses empty {})
                                        layoutEntry.texture.sampleType = WGPUTextureSampleType_Float;
                                    } else if (sampleType == "unfilterable-float") {
                                        layoutEntry.texture.sampleType = WGPUTextureSampleType_UnfilterableFloat;
                                    } else if (sampleType == "depth") {
                                        layoutEntry.texture.sampleType = WGPUTextureSampleType_Depth;
                                    } else if (sampleType == "sint") {
                                        layoutEntry.texture.sampleType = WGPUTextureSampleType_Sint;
                                    } else if (sampleType == "uint") {
                                        layoutEntry.texture.sampleType = WGPUTextureSampleType_Uint;
                                    } else {
                                        // Default to float for unknown types
                                        layoutEntry.texture.sampleType = WGPUTextureSampleType_Float;
                                    }

                                    auto viewDim = g_engine->getProperty(texture, "viewDimension");
                                    if (!g_engine->isUndefined(viewDim)) {
                                        layoutEntry.texture.viewDimension = stringToTextureViewDimension(g_engine->toString(viewDim));
                                    } else {
                                        layoutEntry.texture.viewDimension = WGPUTextureViewDimension_2D;
                                    }

                                    auto multisampled = g_engine->getProperty(texture, "multisampled");
                                    layoutEntry.texture.multisampled = !g_engine->isUndefined(multisampled) && g_engine->toBoolean(multisampled);
                                }

                                // Check for storageTexture binding
                                auto storageTexture = g_engine->getProperty(entry, "storageTexture");
                                if (!g_engine->isUndefined(storageTexture)) {
                                    std::string access = g_engine->toString(g_engine->getProperty(storageTexture, "access"));
                                    if (access == "write-only") {
                                        layoutEntry.storageTexture.access = WGPUStorageTextureAccess_WriteOnly;
                                    } else if (access == "read-only") {
                                        layoutEntry.storageTexture.access = WGPUStorageTextureAccess_ReadOnly;
                                    } else if (access == "read-write") {
                                        layoutEntry.storageTexture.access = WGPUStorageTextureAccess_ReadWrite;
                                    }

                                    auto format = g_engine->getProperty(storageTexture, "format");
                                    if (!g_engine->isUndefined(format)) {
                                        layoutEntry.storageTexture.format = stringToFormat(g_engine->toString(format));
                                    }

                                    auto viewDim = g_engine->getProperty(storageTexture, "viewDimension");
                                    if (!g_engine->isUndefined(viewDim)) {
                                        layoutEntry.storageTexture.viewDimension = stringToTextureViewDimension(g_engine->toString(viewDim));
                                    } else {
                                        layoutEntry.storageTexture.viewDimension = WGPUTextureViewDimension_2D;
                                    }
                                }

                                // Check for external texture binding (WebGPU chain extension)
                                auto externalTexture = g_engine->getProperty(entry, "externalTexture");
                                if (!g_engine->isUndefined(externalTexture)) {
                                    externalTextureLayouts.emplace_back();
                                    auto& externalLayout = externalTextureLayouts.back();
                                    externalLayout.chain.next = nullptr;
                                    externalLayout.chain.sType = WGPUSType_ExternalTextureBindingLayout;
                                    layoutEntry.nextInChain = &externalLayout.chain;
                                }

                                layoutEntries.push_back(layoutEntry);
                            }

                            WGPUBindGroupLayoutDescriptor layoutDesc = {};
                            layoutDesc.entryCount = layoutEntries.size();
                            layoutDesc.entries = layoutEntries.data();

                            WGPUBindGroupLayout layout = wgpuDeviceCreateBindGroupLayout(g_device, &layoutDesc);

                            auto jsLayout = g_engine->newObject();
                            manageResource(jsLayout, layout, ResourceKind::BindGroupLayout);

                            if (g_verboseLogging) std::cout << "[WebGPU] Created bind group layout with " << entryCount << " entries" << std::endl;
                            return jsLayout;
                        })
                    );

                    // device.createBindGroup(descriptor)
                    g_engine->setProperty(device, "createBindGroup",
                        g_engine->newFunction("createBindGroup", [](void* ctx, const std::vector<js::JSValueHandle>& args) {
                            if (args.empty()) {
                                g_engine->throwException("createBindGroup requires a descriptor");
                                return g_engine->newUndefined();
                            }

                            auto descriptor = args[0];
                            auto layoutHandle = g_engine->getProperty(descriptor, "layout");
                            WGPUBindGroupLayout layout = (WGPUBindGroupLayout)resourceData(layoutHandle);

                            auto entries = g_engine->getProperty(descriptor, "entries");
                            if (!sequenceArray(entries, entries)) return g_engine->newUndefined();
                            auto lengthProp = g_engine->getProperty(entries, "length");
                            int entryCount = g_engine->isUndefined(lengthProp) ? 0 : (int)g_engine->toNumber(lengthProp);

                            std::vector<WGPUBindGroupEntry> bindGroupEntries;
                            bindGroupEntries.reserve(entryCount);
                            std::vector<WGPUTextureView> autoCreatedViews;

                            for (int i = 0; i < entryCount; i++) {
                                auto entry = g_engine->getPropertyIndex(entries, i);

                                WGPUBindGroupEntry bgEntry = {};
                                bgEntry.binding = (uint32_t)g_engine->toNumber(g_engine->getProperty(entry, "binding"));

                                auto resource = g_engine->getProperty(entry, "resource");

                                // Check if resource is a sampler (has no buffer property)
                                auto bufferProp = g_engine->getProperty(resource, "buffer");
                                if (!g_engine->isUndefined(bufferProp)) {
                                    // Buffer binding: {buffer, offset?, size?}
                                    bgEntry.buffer = (WGPUBuffer)resourceData(bufferProp);

                                    auto offset = g_engine->getProperty(resource, "offset");
                                    bgEntry.offset = g_engine->isUndefined(offset) ? 0 : (uint64_t)g_engine->toNumber(offset);

                                    auto size = g_engine->getProperty(resource, "size");
                                    // Size 0 means whole buffer
                                    bgEntry.size = g_engine->isUndefined(size) ? WGPU_WHOLE_SIZE : (uint64_t)g_engine->toNumber(size);
                                } else {
                                    // Could be a sampler or texture view
                                    void* resourcePtr = resourceData(resource);

                                    // Check for type hints set when creating the object
                                    auto typeHint = g_engine->getProperty(resource, "_type");
                                    if (!g_engine->isUndefined(typeHint)) {
                                        std::string typeStr = g_engine->toString(typeHint);
                                        if (typeStr == "sampler") {
                                            if (resourcePtr) {
                                                bgEntry.sampler = (WGPUSampler)resourcePtr;
                                            } else {
                                                std::cerr << "[WebGPU] Warning: Sampler at binding " << bgEntry.binding << " is null" << std::endl;
                                            }
                                        } else if (typeStr == "textureView") {
                                            if (resourcePtr) {
                                                bgEntry.textureView = (WGPUTextureView)resourcePtr;
                                            } else {
                                                std::cerr << "[WebGPU] Warning: TextureView at binding " << bgEntry.binding << " is null, creating placeholder" << std::endl;
                                                // Create a 1x1 placeholder texture view to avoid validation errors
                                                // This is a workaround for textures that failed to create
                                            }
                                        }
                                    } else if (resourcePtr) {
                                        // No type hint - try to detect based on properties
                                        // Check if it looks like a texture (has width/height/format properties)
                                        auto widthProp = g_engine->getProperty(resource, "width");
                                        auto formatProp = g_engine->getProperty(resource, "format");
                                        if (!g_engine->isUndefined(widthProp) && !g_engine->isUndefined(formatProp)) {
                                            // This is a texture, create a view automatically
                                            WGPUTexture tex = (WGPUTexture)resourcePtr;
                                            WGPUTextureViewDescriptor viewDesc = {};
                                            WGPUTextureView view = wgpuTextureCreateView(tex, &viewDesc);

                                            autoCreatedViews.push_back(view);
                                            bgEntry.textureView = view;
                                            if (g_verboseLogging) std::cout << "[WebGPU] Auto-created texture view for binding " << bgEntry.binding << std::endl;
                                        } else {
                                            // Assume sampler as fallback
                                            bgEntry.sampler = (WGPUSampler)resourcePtr;
                                        }
                                    } else {
                                        std::cerr << "[WebGPU] Warning: Resource at binding " << bgEntry.binding << " has null privateData" << std::endl;
                                    }
                                }

                                bindGroupEntries.push_back(bgEntry);
                            }

                            WGPUBindGroupDescriptor bgDesc = {};
                            bgDesc.layout = layout;
                            bgDesc.entryCount = bindGroupEntries.size();
                            bgDesc.entries = bindGroupEntries.data();

                            WGPUBindGroup bindGroup = wgpuDeviceCreateBindGroup(g_device, &bgDesc);

                            // Release auto-created texture views — Dawn holds its own
                            // internal references through the bind group
                            for (auto v : autoCreatedViews) {
                                wgpuTextureViewRelease(v);
                            }

                            auto jsBindGroup = g_engine->newObject();
                            manageResource(jsBindGroup, bindGroup, ResourceKind::BindGroup);


                            if (g_verboseLogging) std::cout << "[WebGPU] Created bind group with " << entryCount << " entries" << std::endl;
                            return jsBindGroup;
                        })
                    );

                    // device.createPipelineLayout(descriptor)
                    g_engine->setProperty(device, "createPipelineLayout",
                        g_engine->newFunction("createPipelineLayout", [](void* ctx, const std::vector<js::JSValueHandle>& args) {
                            if (args.empty()) {
                                g_engine->throwException("createPipelineLayout requires a descriptor");
                                return g_engine->newUndefined();
                            }

                            auto descriptor = args[0];
                            auto bindGroupLayouts = g_engine->getProperty(descriptor, "bindGroupLayouts");
                            if (!sequenceArray(bindGroupLayouts, bindGroupLayouts)) return g_engine->newUndefined();
                            auto lengthProp = g_engine->getProperty(bindGroupLayouts, "length");
                            int layoutCount = g_engine->isUndefined(lengthProp) ? 0 : (int)g_engine->toNumber(lengthProp);

                            std::vector<WGPUBindGroupLayout> layouts;
                            layouts.reserve(layoutCount);

                            for (int i = 0; i < layoutCount; i++) {
                                auto layoutHandle = g_engine->getPropertyIndex(bindGroupLayouts, i);
                                WGPUBindGroupLayout layout = (WGPUBindGroupLayout)resourceData(layoutHandle);
                                layouts.push_back(layout);
                            }

                            WGPUPipelineLayoutDescriptor layoutDesc = {};
                            layoutDesc.bindGroupLayoutCount = layouts.size();
                            layoutDesc.bindGroupLayouts = layouts.data();

                            WGPUPipelineLayout pipelineLayout = wgpuDeviceCreatePipelineLayout(g_device, &layoutDesc);

                            auto jsLayout = g_engine->newObject();
                            manageResource(jsLayout, pipelineLayout, ResourceKind::PipelineLayout);

                            if (g_verboseLogging) std::cout << "[WebGPU] Created pipeline layout with " << layoutCount << " bind group layouts" << std::endl;
                            return jsLayout;
                        })
                    );

                    // device.createTextureView(texture, descriptor?) - Non-standard helper
                    // Workaround because texture.createView() can't easily access 'this'
                    g_engine->setProperty(device, "createTextureView",
                        g_engine->newFunction("createTextureView", [](void* ctx, const std::vector<js::JSValueHandle>& args) {
                            if (args.empty()) {
                                g_engine->throwException("createTextureView requires a texture");
                                return g_engine->newUndefined();
                            }

                            auto textureHandle = args[0];
                            WGPUTexture texture = (WGPUTexture)resourceData(textureHandle);

                            if (!texture) {
                                g_engine->throwException("createTextureView: invalid texture");
                                return g_engine->newUndefined();
                            }

                            // Get texture info
                            double formatEnum = g_engine->toNumber(g_engine->getProperty(textureHandle, "_formatEnum"));
                            WGPUTextureFormat format = formatEnum == 0 ? g_surfaceFormat : (WGPUTextureFormat)(int)formatEnum;

                            // Get format from _textureId if available
                            auto textureIdVal = g_engine->getProperty(textureHandle, "_textureId");
                            if (!g_engine->isUndefined(textureIdVal)) {
                                uint64_t textureId = (uint64_t)g_engine->toNumber(textureIdVal);
                                auto it = g_textureRegistry.find(textureId);
                                if (it != g_textureRegistry.end()) {
                                    format = it->second.format;
                                }
                            }

                            WGPUTextureViewDescriptor viewDesc = {};
                            viewDesc.format = format;
                            viewDesc.dimension = WGPUTextureViewDimension_2D;
                            viewDesc.baseMipLevel = 0;
                            viewDesc.mipLevelCount = 1;
                            viewDesc.baseArrayLayer = 0;
                            viewDesc.arrayLayerCount = 1;
                            viewDesc.aspect = WGPUTextureAspect_All;

                            // Parse descriptor if provided
                            if (args.size() > 1 && !g_engine->isUndefined(args[1])) {
                                auto descriptor = args[1];

                                auto formatProp = g_engine->getProperty(descriptor, "format");
                                if (!g_engine->isUndefined(formatProp)) {
                                    viewDesc.format = stringToFormat(g_engine->toString(formatProp));
                                }

                                auto dimensionProp = g_engine->getProperty(descriptor, "dimension");
                                if (!g_engine->isUndefined(dimensionProp)) {
                                    viewDesc.dimension = stringToTextureViewDimension(g_engine->toString(dimensionProp));
                                }

                                auto baseMipLevel = g_engine->getProperty(descriptor, "baseMipLevel");
                                if (!g_engine->isUndefined(baseMipLevel)) {
                                    viewDesc.baseMipLevel = (uint32_t)g_engine->toNumber(baseMipLevel);
                                }

                                auto mipLevelCount = g_engine->getProperty(descriptor, "mipLevelCount");
                                if (!g_engine->isUndefined(mipLevelCount)) {
                                    viewDesc.mipLevelCount = (uint32_t)g_engine->toNumber(mipLevelCount);
                                }

                                auto baseArrayLayer = g_engine->getProperty(descriptor, "baseArrayLayer");
                                if (!g_engine->isUndefined(baseArrayLayer)) {
                                    viewDesc.baseArrayLayer = (uint32_t)g_engine->toNumber(baseArrayLayer);
                                }

                                auto arrayLayerCount = g_engine->getProperty(descriptor, "arrayLayerCount");
                                if (!g_engine->isUndefined(arrayLayerCount)) {
                                    uint32_t requested = (uint32_t)g_engine->toNumber(arrayLayerCount);
                                    // Clamp to 1 for surface textures (which only have 1 layer)
                                    // or look up actual layer count from registry
                                    auto textureIdVal2 = g_engine->getProperty(textureHandle, "_textureId");
                                    uint32_t maxLayers = 1;
                                    if (!g_engine->isUndefined(textureIdVal2)) {
                                        uint64_t tid = (uint64_t)g_engine->toNumber(textureIdVal2);
                                        auto it = g_textureRegistry.find(tid);
                                        if (it != g_textureRegistry.end()) {
                                            maxLayers = it->second.depthOrArrayLayers > 0 ? it->second.depthOrArrayLayers : 1;
                                        }
                                    }
                                    viewDesc.arrayLayerCount = std::min(requested, maxLayers - viewDesc.baseArrayLayer);
                                }

                                auto aspect = g_engine->getProperty(descriptor, "aspect");
                                if (!g_engine->isUndefined(aspect)) {
                                    std::string aspectStr = g_engine->toString(aspect);
                                    if (aspectStr == "all") viewDesc.aspect = WGPUTextureAspect_All;
                                    else if (aspectStr == "stencil-only") viewDesc.aspect = WGPUTextureAspect_StencilOnly;
                                    else if (aspectStr == "depth-only") viewDesc.aspect = WGPUTextureAspect_DepthOnly;
                                }
                            }

                            // Final validation: Fix arrayLayerCount based on view dimension
                            if (viewDesc.dimension == WGPUTextureViewDimension_3D ||
                                viewDesc.dimension == WGPUTextureViewDimension_1D) {
                                viewDesc.arrayLayerCount = 1;
                            } else if (viewDesc.dimension == WGPUTextureViewDimension_Cube) {
                                viewDesc.arrayLayerCount = 6;
                            }

                            WGPUTextureView view = wgpuTextureCreateView(texture, &viewDesc);

                            auto jsView = g_engine->newObject();
                            manageTextureView(jsView, view, texture);
                            g_engine->setProperty(jsView, "_type", g_engine->newString("textureView"));


                            if (g_verboseLogging) std::cout << "[WebGPU] Created texture view" << std::endl;
                            return jsView;
                        })
                    );

                    // device.createRenderBundleEncoder(descriptor)
                    // Used by Three.js for mipmap generation
                    g_engine->setProperty(device, "createRenderBundleEncoder",
                        g_engine->newFunction("createRenderBundleEncoder", [](void* ctx, const std::vector<js::JSValueHandle>& args) {
                            if (args.empty()) {
                                g_engine->throwException("createRenderBundleEncoder requires a descriptor");
                                return g_engine->newUndefined();
                            }

                            auto descriptor = args[0];

                            // Parse color formats
                            auto colorFormats = g_engine->getProperty(descriptor, "colorFormats");
                            auto colorFormatsLength = g_engine->getProperty(colorFormats, "length");
                            int colorFormatCount = g_engine->isUndefined(colorFormatsLength) ? 0 : (int)g_engine->toNumber(colorFormatsLength);

                            std::vector<WGPUTextureFormat> formats;
                            formats.reserve(colorFormatCount);
                            for (int i = 0; i < colorFormatCount; i++) {
                                auto formatProp = g_engine->getPropertyIndex(colorFormats, i);
                                if (!g_engine->isUndefined(formatProp) && !g_engine->isNull(formatProp)) {
                                    formats.push_back(stringToFormat(g_engine->toString(formatProp)));
                                }
                            }

                            // Parse depth stencil format
                            WGPUTextureFormat depthFormat = WGPUTextureFormat_Undefined;
                            auto depthFormatProp = g_engine->getProperty(descriptor, "depthStencilFormat");
                            if (!g_engine->isUndefined(depthFormatProp) && !g_engine->isNull(depthFormatProp)) {
                                depthFormat = stringToFormat(g_engine->toString(depthFormatProp));
                            }

                            // Parse sample count
                            uint32_t sampleCount = 1;
                            auto sampleCountProp = g_engine->getProperty(descriptor, "sampleCount");
                            if (!g_engine->isUndefined(sampleCountProp)) {
                                sampleCount = (uint32_t)g_engine->toNumber(sampleCountProp);
                            }

                            WGPURenderBundleEncoderDescriptor desc = {};
                            desc.colorFormatCount = formats.size();
                            desc.colorFormats = formats.data();
                            desc.depthStencilFormat = depthFormat;
                            desc.sampleCount = sampleCount;

                            WGPURenderBundleEncoder bundleEncoder = wgpuDeviceCreateRenderBundleEncoder(g_device, &desc);

                            if (!bundleEncoder) {
                                g_engine->throwException("Failed to create render bundle encoder");
                                return g_engine->newUndefined();
                            }

                            auto jsEncoder = g_engine->newObject();
                            const auto encoderResourceId = manageResource(jsEncoder, bundleEncoder, ResourceKind::RenderBundleEncoder);

                            // Capture for closures
                            WGPURenderBundleEncoder capturedEncoder = bundleEncoder;

                            // renderBundleEncoder.setPipeline(pipeline)
                            g_engine->setProperty(jsEncoder, "setPipeline",
                                g_engine->newFunction("setPipeline", [capturedEncoder, encoderResourceId](void* ctx, const std::vector<js::JSValueHandle>& args) {
                                            auto activeResource = resourceHandle<WGPURenderBundleEncoder>(encoderResourceId);
                                            if (!activeResource) return g_engine->newUndefined();
                                    if (args.empty()) return g_engine->newUndefined();
                                    WGPURenderPipeline pipeline = (WGPURenderPipeline)resourceData(args[0]);
                                    wgpuRenderBundleEncoderSetPipeline(capturedEncoder, pipeline);
                                    return g_engine->newUndefined();
                                })
                            );

                            // renderBundleEncoder.setVertexBuffer(slot, buffer, offset?, size?)
                            g_engine->setProperty(jsEncoder, "setVertexBuffer",
                                g_engine->newFunction("setVertexBuffer", [capturedEncoder, encoderResourceId](void* ctx, const std::vector<js::JSValueHandle>& args) {
                                            auto activeResource = resourceHandle<WGPURenderBundleEncoder>(encoderResourceId);
                                            if (!activeResource) return g_engine->newUndefined();
                                    if (args.size() < 2) return g_engine->newUndefined();
                                    uint32_t slot = (uint32_t)g_engine->toNumber(args[0]);
                                    WGPUBuffer buffer = (WGPUBuffer)resourceData(args[1]);
                                    uint64_t offset = args.size() > 2 && !g_engine->isUndefined(args[2]) ? (uint64_t)g_engine->toNumber(args[2]) : 0;
                                    uint64_t size = args.size() > 3 && !g_engine->isUndefined(args[3]) ? (uint64_t)g_engine->toNumber(args[3]) : WGPU_WHOLE_SIZE;
                                    wgpuRenderBundleEncoderSetVertexBuffer(capturedEncoder, slot, buffer, offset, size);
                                    return g_engine->newUndefined();
                                })
                            );

                            // renderBundleEncoder.setIndexBuffer(buffer, format, offset?, size?)
                            g_engine->setProperty(jsEncoder, "setIndexBuffer",
                                g_engine->newFunction("setIndexBuffer", [capturedEncoder, encoderResourceId](void* ctx, const std::vector<js::JSValueHandle>& args) {
                                            auto activeResource = resourceHandle<WGPURenderBundleEncoder>(encoderResourceId);
                                            if (!activeResource) return g_engine->newUndefined();
                                    if (args.size() < 2) return g_engine->newUndefined();
                                    WGPUBuffer buffer = (WGPUBuffer)resourceData(args[0]);
                                    std::string formatStr = g_engine->toString(args[1]);
                                    WGPUIndexFormat format = formatStr == "uint32" ? WGPUIndexFormat_Uint32 : WGPUIndexFormat_Uint16;
                                    uint64_t offset = args.size() > 2 && !g_engine->isUndefined(args[2]) ? (uint64_t)g_engine->toNumber(args[2]) : 0;
                                    uint64_t size = args.size() > 3 && !g_engine->isUndefined(args[3]) ? (uint64_t)g_engine->toNumber(args[3]) : WGPU_WHOLE_SIZE;
                                    wgpuRenderBundleEncoderSetIndexBuffer(capturedEncoder, buffer, format, offset, size);
                                    return g_engine->newUndefined();
                                })
                            );

                            // renderBundleEncoder.setBindGroup(index, bindGroup, dynamicOffsets?)
                            g_engine->setProperty(jsEncoder, "setBindGroup",
                                g_engine->newFunction("setBindGroup", [capturedEncoder, encoderResourceId](void* ctx, const std::vector<js::JSValueHandle>& args) {
                                            auto activeResource = resourceHandle<WGPURenderBundleEncoder>(encoderResourceId);
                                            if (!activeResource) return g_engine->newUndefined();
                                    if (args.size() < 2) return g_engine->newUndefined();
                                    uint32_t index = (uint32_t)g_engine->toNumber(args[0]);
                                    WGPUBindGroup bindGroup = g_engine->isNull(args[1]) || g_engine->isUndefined(args[1]) ?
                                        nullptr : static_cast<WGPUBindGroup>(resourceData(args[1]));
                                    std::vector<uint32_t> offsets;
                                    if (!dynamicOffsets(args, offsets)) return g_engine->newUndefined();
                                    wgpuRenderBundleEncoderSetBindGroup(capturedEncoder, index, bindGroup, offsets.size(), offsets.data());
                                    return g_engine->newUndefined();
                                })
                            );

                            // renderBundleEncoder.draw(vertexCount, instanceCount?, firstVertex?, firstInstance?)
                            g_engine->setProperty(jsEncoder, "draw",
                                g_engine->newFunction("draw", [capturedEncoder, encoderResourceId](void* ctx, const std::vector<js::JSValueHandle>& args) {
                                            auto activeResource = resourceHandle<WGPURenderBundleEncoder>(encoderResourceId);
                                            if (!activeResource) return g_engine->newUndefined();
                                    if (args.empty()) return g_engine->newUndefined();
                                    uint32_t vertexCount = (uint32_t)g_engine->toNumber(args[0]);
                                    uint32_t instanceCount = args.size() > 1 ? (uint32_t)g_engine->toNumber(args[1]) : 1;
                                    uint32_t firstVertex = args.size() > 2 ? (uint32_t)g_engine->toNumber(args[2]) : 0;
                                    uint32_t firstInstance = args.size() > 3 ? (uint32_t)g_engine->toNumber(args[3]) : 0;
                                    wgpuRenderBundleEncoderDraw(capturedEncoder, vertexCount, instanceCount, firstVertex, firstInstance);
                                    return g_engine->newUndefined();
                                })
                            );

                            // renderBundleEncoder.drawIndexed(indexCount, instanceCount?, firstIndex?, baseVertex?, firstInstance?)
                            g_engine->setProperty(jsEncoder, "drawIndexed",
                                g_engine->newFunction("drawIndexed", [capturedEncoder, encoderResourceId](void* ctx, const std::vector<js::JSValueHandle>& args) {
                                            auto activeResource = resourceHandle<WGPURenderBundleEncoder>(encoderResourceId);
                                            if (!activeResource) return g_engine->newUndefined();
                                    if (args.empty()) return g_engine->newUndefined();
                                    uint32_t indexCount = (uint32_t)g_engine->toNumber(args[0]);
                                    uint32_t instanceCount = args.size() > 1 ? (uint32_t)g_engine->toNumber(args[1]) : 1;
                                    uint32_t firstIndex = args.size() > 2 ? (uint32_t)g_engine->toNumber(args[2]) : 0;
                                    int32_t baseVertex = args.size() > 3 ? (int32_t)g_engine->toNumber(args[3]) : 0;
                                    uint32_t firstInstance = args.size() > 4 ? (uint32_t)g_engine->toNumber(args[4]) : 0;
                                    wgpuRenderBundleEncoderDrawIndexed(capturedEncoder, indexCount, instanceCount, firstIndex, baseVertex, firstInstance);
                                    return g_engine->newUndefined();
                                })
                            );

                            // renderBundleEncoder.finish(descriptor?)
                            g_engine->setProperty(jsEncoder, "finish",
                                g_engine->newFunction("finish", [capturedEncoder, encoderResourceId](void* ctx, const std::vector<js::JSValueHandle>& args) {
                                            auto activeResource = resourceHandle<WGPURenderBundleEncoder>(encoderResourceId);
                                            if (!activeResource) return g_engine->newUndefined();
                                    WGPURenderBundleDescriptor desc = {};
                                    WGPURenderBundle bundle = wgpuRenderBundleEncoderFinish(capturedEncoder, &desc);
                                    releaseResource(encoderResourceId);

                                    auto jsBundle = g_engine->newObject();
                                    manageResource(jsBundle, bundle, ResourceKind::RenderBundle);
                                    g_engine->setProperty(jsBundle, "_type", g_engine->newString("renderBundle"));

                                    if (g_verboseLogging) std::cout << "[WebGPU] Render bundle finished" << std::endl;
                                    return jsBundle;
                                })
                            );

                            if (g_verboseLogging) std::cout << "[WebGPU] Created render bundle encoder" << std::endl;
                            return jsEncoder;
                        })
                    );

                    // device.pushErrorScope(filter) - Push an error scope for validation/OOM/internal errors
                    // Used by Three.js for error handling during pipeline creation
                    g_engine->setProperty(device, "pushErrorScope",
                        g_engine->newFunction("pushErrorScope", [](void* ctx, const std::vector<js::JSValueHandle>& args) {
                            const std::string filter = args.empty() ? "" : g_engine->toString(args[0]);
                            WGPUErrorFilter native;
                            if (filter == "validation") native = WGPUErrorFilter_Validation;
                            else if (filter == "out-of-memory") native = WGPUErrorFilter_OutOfMemory;
                            else if (filter == "internal") native = WGPUErrorFilter_Internal;
                            else {
                                g_engine->throwException("Invalid WebGPU error scope filter");
                                return g_engine->newUndefined();
                            }
                            wgpuDevicePushErrorScope(g_device, native);
                            return g_engine->newUndefined();
                        })
                    );

                    // device.popErrorScope() - Pop an error scope and return Promise<GPUError | null>
                    // Returns Promise<GPUError | null>
                    g_engine->setProperty(device, "popErrorScope",
                        g_engine->newFunction("popErrorScope", [](void* ctx, const std::vector<js::JSValueHandle>& args) {
                            auto deferred = makeDeferred();
#if defined(MYSTRAL_WEBGPU_DAWN)
                            WGPUPopErrorScopeCallbackInfo callback = {};
                            callback.mode = WGPUCallbackMode_AllowProcessEvents;
                            callback.callback = onScopeCompleted;
                            callback.userdata1 = new std::shared_ptr<DeferredResult>(deferred);
                            wgpuDevicePopErrorScope(g_device, callback);
#else
                            settle(deferred, false, nativeError(WGPUErrorType_Internal, "Native error scope binding is unavailable for this backend"));
#endif
                            return deferred->promise;
                        })
                    );

                    g_lostResult = makeDeferred();
                    g_engine->setProperty(device, "lost", g_lostResult->promise);
                    g_engine->setProperty(device, "destroy",
                        g_engine->newFunction("destroy", [](void*, const std::vector<js::JSValueHandle>&) {
                            if (!g_deviceDestroyed) {
                                const bool retired = releaseContextSurfaceBeforeDeviceDestroy(g_device);
                                if (g_surface && !retired) {
                                    g_engine->throwException("Cannot retire the device's Context-owned surface");
                                    return g_engine->newUndefined();
                                }
                                cleanupOwnedResources();
                                g_deviceDestroyed = true;
                                wgpuDeviceDestroy(g_device);
                            }
                            return g_engine->newUndefined();
                        }));
                    g_jsDevice = device;
                    g_engine->protect(g_jsDevice);

                    return resolved(device);
                })
            );

            // adapter.features - Set-like object that is also iterable
            // We use an array for iteration support with a has() method added
            // Dawn supports indirect-first-instance on Metal which is required for indirect draws
            // with non-zero firstInstance values
            auto features = nativeFeatures(true);
            g_engine->setProperty(adapter, "features", features);

            // adapter.limits
            auto limits = nativeLimits(true);
            g_engine->setProperty(adapter, "limits", limits);
            WGPUAdapterInfo nativeInfo = {};
#if defined(MYSTRAL_WEBGPU_DAWN)
            if (wgpuAdapterGetInfo(g_adapter, &nativeInfo) != WGPUStatus_Success) {
                g_engine->throwException("Native WebGPU adapter information query failed");
                return g_engine->newUndefined();
            }
#else
            wgpuAdapterGetInfo(g_adapter, &nativeInfo);
#endif
            auto info = g_engine->newObject();
#define EXPOSE_ADAPTER_INFO(name) g_engine->setProperty(info, #name, g_engine->newString(copyNativeString(nativeInfo.name).c_str()))
            EXPOSE_ADAPTER_INFO(vendor); EXPOSE_ADAPTER_INFO(architecture);
            EXPOSE_ADAPTER_INFO(device); EXPOSE_ADAPTER_INFO(description);
#undef EXPOSE_ADAPTER_INFO
            wgpuAdapterInfoFreeMembers(nativeInfo);
            g_engine->setProperty(adapter, "info", info);

            return resolved(adapter);
        })
    );

    // navigator.gpu.getPreferredCanvasFormat()
    engine->setProperty(gpuObject, "getPreferredCanvasFormat",
        engine->newFunction("getPreferredCanvasFormat", [](void* ctx, const std::vector<js::JSValueHandle>& args) {
            return g_engine->newString(formatToString(g_surfaceFormat));
        })
    );

    // Set navigator.gpu
    engine->setProperty(navigatorHandle, "gpu", gpuObject);

    // ========================================================================
    // GPU Usage Constants
    // ========================================================================
    auto gpuBufferUsage = engine->newObject();
    engine->setProperty(gpuBufferUsage, "MAP_READ", engine->newNumber(0x0001));
    engine->setProperty(gpuBufferUsage, "MAP_WRITE", engine->newNumber(0x0002));
    engine->setProperty(gpuBufferUsage, "COPY_SRC", engine->newNumber(0x0004));
    engine->setProperty(gpuBufferUsage, "COPY_DST", engine->newNumber(0x0008));
    engine->setProperty(gpuBufferUsage, "INDEX", engine->newNumber(0x0010));
    engine->setProperty(gpuBufferUsage, "VERTEX", engine->newNumber(0x0020));
    engine->setProperty(gpuBufferUsage, "UNIFORM", engine->newNumber(0x0040));
    engine->setProperty(gpuBufferUsage, "STORAGE", engine->newNumber(0x0080));
    engine->setProperty(gpuBufferUsage, "INDIRECT", engine->newNumber(0x0100));
    engine->setProperty(gpuBufferUsage, "QUERY_RESOLVE", engine->newNumber(0x0200));
    engine->setGlobalProperty("GPUBufferUsage", gpuBufferUsage);

    auto gpuTextureUsage = engine->newObject();
    engine->setProperty(gpuTextureUsage, "COPY_SRC", engine->newNumber(0x01));
    engine->setProperty(gpuTextureUsage, "COPY_DST", engine->newNumber(0x02));
    engine->setProperty(gpuTextureUsage, "TEXTURE_BINDING", engine->newNumber(0x04));
    engine->setProperty(gpuTextureUsage, "STORAGE_BINDING", engine->newNumber(0x08));
    engine->setProperty(gpuTextureUsage, "RENDER_ATTACHMENT", engine->newNumber(0x10));
    engine->setGlobalProperty("GPUTextureUsage", gpuTextureUsage);

    auto gpuShaderStage = engine->newObject();
    engine->setProperty(gpuShaderStage, "VERTEX", engine->newNumber(0x1));
    engine->setProperty(gpuShaderStage, "FRAGMENT", engine->newNumber(0x2));
    engine->setProperty(gpuShaderStage, "COMPUTE", engine->newNumber(0x4));
    engine->setGlobalProperty("GPUShaderStage", gpuShaderStage);

    auto gpuMapMode = engine->newObject();
    engine->setProperty(gpuMapMode, "READ", engine->newNumber(0x1));
    engine->setProperty(gpuMapMode, "WRITE", engine->newNumber(0x2));
    engine->setGlobalProperty("GPUMapMode", gpuMapMode);

    auto gpuColorWrite = engine->newObject();
    engine->setProperty(gpuColorWrite, "RED", engine->newNumber(WGPUColorWriteMask_Red));
    engine->setProperty(gpuColorWrite, "GREEN", engine->newNumber(WGPUColorWriteMask_Green));
    engine->setProperty(gpuColorWrite, "BLUE", engine->newNumber(WGPUColorWriteMask_Blue));
    engine->setProperty(gpuColorWrite, "ALPHA", engine->newNumber(WGPUColorWriteMask_Alpha));
    engine->setProperty(gpuColorWrite, "ALL", engine->newNumber(WGPUColorWriteMask_All));
    engine->setGlobalProperty("GPUColorWrite", gpuColorWrite);

    // =========================================================================
    // createImageBitmap() - Standard Web API for image decoding
    // =========================================================================
    // createImageBitmap(source) -> Promise<ImageBitmap>
    // source can be: Blob, ArrayBuffer, or object with arrayBuffer() method
    // Returns ImageBitmap with: width, height, close(), and internal pixel data
    //
    // Note: PNG/JPEG supported via stb_image. WebP supported via libwebp (when MYSTRAL_HAS_WEBP defined).

    // Native helper that decodes image data synchronously
    engine->setGlobalProperty("__decodeImageData",
        engine->newFunction("__decodeImageData", [](void* ctx, const std::vector<js::JSValueHandle>& args) {
            if (args.empty()) {
                g_engine->throwException("__decodeImageData requires an ArrayBuffer argument");
                return g_engine->newUndefined();
            }

            // Get ArrayBuffer data
            size_t inputSize = 0;
            void* inputData = g_engine->getArrayBufferData(args[0], &inputSize);

            if (!inputData || inputSize == 0) {
                g_engine->throwException("__decodeImageData: invalid ArrayBuffer");
                return g_engine->newUndefined();
            }

            const unsigned char* inputBytes = (const unsigned char*)inputData;
            int width = 0, height = 0;
            unsigned char* data = nullptr;
            bool isWebP = false;

            // Check if this is a WebP image (starts with "RIFF" and has "WEBP" at offset 8)
            if (inputSize >= 12 &&
                inputBytes[0] == 'R' && inputBytes[1] == 'I' &&
                inputBytes[2] == 'F' && inputBytes[3] == 'F' &&
                inputBytes[8] == 'W' && inputBytes[9] == 'E' &&
                inputBytes[10] == 'B' && inputBytes[11] == 'P') {
                isWebP = true;
            }

            if (isWebP) {
#ifdef MYSTRAL_HAS_WEBP
                // Decode WebP using libwebp
                data = WebPDecodeRGBA(inputBytes, inputSize, &width, &height);
                if (!data) {
                    g_engine->throwException("Failed to decode WebP image");
                    return g_engine->newUndefined();
                }
                if (g_verboseLogging) std::cout << "[createImageBitmap] Decoded WebP " << width << "x" << height << " image" << std::endl;
#else
                g_engine->throwException("WebP image detected but libwebp support not compiled in. Rebuild with MYSTRAL_HAS_WEBP.");
                return g_engine->newUndefined();
#endif
            } else {
                // Decode using stb_image (PNG, JPEG, etc.)
                int channels;
                data = stbi_load_from_memory(inputBytes, (int)inputSize, &width, &height, &channels, 4);
                if (!data) {
                    std::string error = std::string("Failed to decode image: ") + stbi_failure_reason();
                    g_engine->throwException(error.c_str());
                    return g_engine->newUndefined();
                }
                if (g_verboseLogging) std::cout << "[createImageBitmap] Decoded " << width << "x" << height << " image" << std::endl;
            }

            // Create ImageBitmap-like object
            auto result = g_engine->newObject();

            // Create ArrayBuffer with RGBA pixel data
            size_t dataSize = width * height * 4;
            auto arrayBuffer = g_engine->newArrayBuffer(data, dataSize);

            g_engine->setProperty(result, "width", g_engine->newNumber(width));
            g_engine->setProperty(result, "height", g_engine->newNumber(height));
            g_engine->setProperty(result, "_data", arrayBuffer);  // Internal pixel data
            g_engine->setProperty(result, "_closed", g_engine->newBoolean(false));

            // Free decoded data (we copied it to ArrayBuffer)
            if (isWebP) {
#ifdef MYSTRAL_HAS_WEBP
                WebPFree(data);
#endif
            } else {
                stbi_image_free(data);
            }

            return result;
        })
    );

    // JavaScript polyfill for createImageBitmap
    const char* imageBitmapPolyfill = R"(
// ImageBitmap class (web-compatible)
class ImageBitmap {
    constructor(width, height, data) {
        this.width = width;
        this.height = height;
        this._data = data;  // Internal RGBA pixel data
        this._closed = false;
    }

    close() {
        this._closed = true;
        this._data = null;
    }
}

// createImageBitmap - Standard Web API
// Supports: Blob, ArrayBuffer, Response, or object with arrayBuffer() method
async function createImageBitmap(source, options) {
    let arrayBuffer;

    if (source instanceof ArrayBuffer) {
        arrayBuffer = source;
    } else if (source instanceof Uint8Array) {
        arrayBuffer = source.buffer;
    } else if (source && typeof source.arrayBuffer === 'function') {
        // Blob or Response
        arrayBuffer = await source.arrayBuffer();
    } else if (source && source._data) {
        // Already an ImageBitmap-like object
        return source;
    } else {
        throw new Error('createImageBitmap: unsupported source type');
    }

    // Decode using native function
    const decoded = __decodeImageData(arrayBuffer);

    if (!decoded) {
        throw new Error('createImageBitmap: failed to decode image');
    }

    // Create ImageBitmap
    const bitmap = new ImageBitmap(decoded.width, decoded.height, decoded._data);
    return bitmap;
}

globalThis.createImageBitmap = createImageBitmap;
globalThis.ImageBitmap = ImageBitmap;

// CanvasRenderingContext2D - Placeholder class for instanceof checks
// The actual implementation is in Canvas2D bindings, this is just for type checking
class CanvasRenderingContext2D {
    constructor() {
        // This constructor is never called directly - contexts are created via getContext('2d')
    }
}
globalThis.CanvasRenderingContext2D = CanvasRenderingContext2D;

// HTMLCanvasElement - Placeholder class for instanceof checks
class HTMLCanvasElement {
    constructor() {}
}
globalThis.HTMLCanvasElement = HTMLCanvasElement;

// OffscreenCanvas - For type checking
class OffscreenCanvas {
    constructor(width, height) {
        this.width = width || 300;
        this.height = height || 150;
        this._contextType = null;
        this._context = null;
    }

    getContext(type, options) {
        if (type === '2d') {
            // For basic 2D context needs
            if (!this._context) {
                this._context = { canvas: this };
            }
            return this._context;
        }
        return null;
    }
}
globalThis.OffscreenCanvas = OffscreenCanvas;
)";
    engine->eval(imageBitmapPolyfill, "imageBitmap-polyfill.js");

    // =========================================================================
    // Mystral.loadGLTF() - GLTF/GLB file loader
    // =========================================================================
    // Returns parsed GLTF data as JavaScript object
    auto mystralNamespace = engine->newObject();

    engine->setProperty(mystralNamespace, "loadGLTF",
        engine->newFunction("loadGLTF", [](void* ctx, const std::vector<js::JSValueHandle>& args) {
            if (args.empty()) {
                g_engine->throwException("loadGLTF requires a file path argument");
                return g_engine->newUndefined();
            }

            std::string path = g_engine->toString(args[0]);
            if (g_verboseLogging) std::cout << "[GLTF] Loading: " << path << std::endl;

            auto gltfData = mystral::gltf::loadGLTF(path);
            if (!gltfData) {
                g_engine->throwException(("Failed to load GLTF file: " + path).c_str());
                return g_engine->newUndefined();
            }

            // Convert to JavaScript object
            auto result = g_engine->newObject();

            // Meshes array
            auto jsMeshes = g_engine->newArray();
            for (size_t mi = 0; mi < gltfData->meshes.size(); mi++) {
                const auto& mesh = gltfData->meshes[mi];
                auto jsMesh = g_engine->newObject();
                g_engine->setProperty(jsMesh, "name", g_engine->newString(mesh.name.c_str()));

                // Primitives array
                auto jsPrimitives = g_engine->newArray();
                for (size_t pi = 0; pi < mesh.primitives.size(); pi++) {
                    const auto& prim = mesh.primitives[pi];
                    auto jsPrim = g_engine->newObject();

                    // Positions Float32Array
                    if (!prim.positions.data.empty()) {
                        auto posArr = g_engine->createFloat32Array(
                            prim.positions.data.data(),
                            prim.positions.data.size()
                        );
                        g_engine->setProperty(jsPrim, "positions", posArr);
                        g_engine->setProperty(jsPrim, "positionCount",
                            g_engine->newNumber((double)prim.positions.count));
                    }

                    // Normals Float32Array
                    if (!prim.normals.data.empty()) {
                        auto normArr = g_engine->createFloat32Array(
                            prim.normals.data.data(),
                            prim.normals.data.size()
                        );
                        g_engine->setProperty(jsPrim, "normals", normArr);
                    }

                    // Texcoords Float32Array
                    if (!prim.texcoords.data.empty()) {
                        auto uvArr = g_engine->createFloat32Array(
                            prim.texcoords.data.data(),
                            prim.texcoords.data.size()
                        );
                        g_engine->setProperty(jsPrim, "texcoords", uvArr);
                    }

                    // Tangents Float32Array
                    if (!prim.tangents.data.empty()) {
                        auto tanArr = g_engine->createFloat32Array(
                            prim.tangents.data.data(),
                            prim.tangents.data.size()
                        );
                        g_engine->setProperty(jsPrim, "tangents", tanArr);
                    }

                    // Indices Uint32Array
                    if (!prim.indices.empty()) {
                        auto idxArr = g_engine->createUint32Array(
                            prim.indices.data(),
                            prim.indices.size()
                        );
                        g_engine->setProperty(jsPrim, "indices", idxArr);
                        g_engine->setProperty(jsPrim, "indexCount",
                            g_engine->newNumber((double)prim.indices.size()));
                    }

                    g_engine->setProperty(jsPrim, "materialIndex",
                        g_engine->newNumber((double)prim.materialIndex));

                    g_engine->setPropertyIndex(jsPrimitives, pi, jsPrim);
                }
                g_engine->setProperty(jsMesh, "primitives", jsPrimitives);
                g_engine->setPropertyIndex(jsMeshes, mi, jsMesh);
            }
            g_engine->setProperty(result, "meshes", jsMeshes);

            // Materials array
            auto jsMaterials = g_engine->newArray();
            for (size_t mi = 0; mi < gltfData->materials.size(); mi++) {
                const auto& mat = gltfData->materials[mi];
                auto jsMat = g_engine->newObject();
                g_engine->setProperty(jsMat, "name", g_engine->newString(mat.name.c_str()));

                // PBR factors
                auto baseColor = g_engine->newArray();
                for (int i = 0; i < 4; i++) {
                    g_engine->setPropertyIndex(baseColor, i, g_engine->newNumber(mat.baseColorFactor[i]));
                }
                g_engine->setProperty(jsMat, "baseColorFactor", baseColor);
                g_engine->setProperty(jsMat, "metallicFactor", g_engine->newNumber(mat.metallicFactor));
                g_engine->setProperty(jsMat, "roughnessFactor", g_engine->newNumber(mat.roughnessFactor));

                // Emissive
                auto emissive = g_engine->newArray();
                for (int i = 0; i < 3; i++) {
                    g_engine->setPropertyIndex(emissive, i, g_engine->newNumber(mat.emissiveFactor[i]));
                }
                g_engine->setProperty(jsMat, "emissiveFactor", emissive);

                // Texture indices
                g_engine->setProperty(jsMat, "baseColorTextureIndex",
                    g_engine->newNumber(mat.baseColorTexture.imageIndex));
                g_engine->setProperty(jsMat, "metallicRoughnessTextureIndex",
                    g_engine->newNumber(mat.metallicRoughnessTexture.imageIndex));
                g_engine->setProperty(jsMat, "normalTextureIndex",
                    g_engine->newNumber(mat.normalTexture.imageIndex));
                g_engine->setProperty(jsMat, "occlusionTextureIndex",
                    g_engine->newNumber(mat.occlusionTexture.imageIndex));
                g_engine->setProperty(jsMat, "emissiveTextureIndex",
                    g_engine->newNumber(mat.emissiveTexture.imageIndex));

                g_engine->setProperty(jsMat, "normalScale", g_engine->newNumber(mat.normalScale));
                g_engine->setProperty(jsMat, "occlusionStrength", g_engine->newNumber(mat.occlusionStrength));
                g_engine->setProperty(jsMat, "alphaCutoff", g_engine->newNumber(mat.alphaCutoff));
                g_engine->setProperty(jsMat, "doubleSided", g_engine->newBoolean(mat.doubleSided));

                const char* alphaModeStr = "OPAQUE";
                if (mat.alphaMode == mystral::gltf::MaterialData::AlphaMode::Mask) alphaModeStr = "MASK";
                else if (mat.alphaMode == mystral::gltf::MaterialData::AlphaMode::Blend) alphaModeStr = "BLEND";
                g_engine->setProperty(jsMat, "alphaMode", g_engine->newString(alphaModeStr));

                g_engine->setPropertyIndex(jsMaterials, mi, jsMat);
            }
            g_engine->setProperty(result, "materials", jsMaterials);

            // Images array (with embedded data as ArrayBuffers)
            auto jsImages = g_engine->newArray();
            for (size_t ii = 0; ii < gltfData->images.size(); ii++) {
                const auto& img = gltfData->images[ii];
                auto jsImg = g_engine->newObject();
                g_engine->setProperty(jsImg, "name", g_engine->newString(img.name.c_str()));
                g_engine->setProperty(jsImg, "uri", g_engine->newString(img.uri.c_str()));
                g_engine->setProperty(jsImg, "mimeType", g_engine->newString(img.mimeType.c_str()));

                // Embedded image data as ArrayBuffer
                if (!img.data.empty()) {
                    auto dataArr = g_engine->createUint8Array(
                        img.data.data(),
                        img.data.size()
                    );
                    g_engine->setProperty(jsImg, "data", dataArr);
                }

                g_engine->setPropertyIndex(jsImages, ii, jsImg);
            }
            g_engine->setProperty(result, "images", jsImages);

            // Nodes array
            auto jsNodes = g_engine->newArray();
            for (size_t ni = 0; ni < gltfData->nodes.size(); ni++) {
                const auto& node = gltfData->nodes[ni];
                auto jsNode = g_engine->newObject();
                g_engine->setProperty(jsNode, "name", g_engine->newString(node.name.c_str()));
                g_engine->setProperty(jsNode, "meshIndex", g_engine->newNumber(node.meshIndex));

                // Transform - store as separate arrays
                auto translation = g_engine->newArray();
                auto rotation = g_engine->newArray();
                auto scale = g_engine->newArray();
                for (int i = 0; i < 3; i++) {
                    g_engine->setPropertyIndex(translation, i, g_engine->newNumber(node.translation[i]));
                    g_engine->setPropertyIndex(scale, i, g_engine->newNumber(node.scale[i]));
                }
                for (int i = 0; i < 4; i++) {
                    g_engine->setPropertyIndex(rotation, i, g_engine->newNumber(node.rotation[i]));
                }
                g_engine->setProperty(jsNode, "translation", translation);
                g_engine->setProperty(jsNode, "rotation", rotation);
                g_engine->setProperty(jsNode, "scale", scale);

                // Matrix (if present)
                if (node.hasMatrix) {
                    auto matrix = g_engine->newArray();
                    for (int i = 0; i < 16; i++) {
                        g_engine->setPropertyIndex(matrix, i, g_engine->newNumber(node.matrix[i]));
                    }
                    g_engine->setProperty(jsNode, "matrix", matrix);
                }

                // Children indices
                auto children = g_engine->newArray();
                for (size_t ci = 0; ci < node.children.size(); ci++) {
                    g_engine->setPropertyIndex(children, ci, g_engine->newNumber(node.children[ci]));
                }
                g_engine->setProperty(jsNode, "children", children);

                g_engine->setPropertyIndex(jsNodes, ni, jsNode);
            }
            g_engine->setProperty(result, "nodes", jsNodes);

            // Scenes array
            auto jsScenes = g_engine->newArray();
            for (size_t si = 0; si < gltfData->scenes.size(); si++) {
                const auto& scene = gltfData->scenes[si];
                auto jsScene = g_engine->newObject();
                g_engine->setProperty(jsScene, "name", g_engine->newString(scene.name.c_str()));

                auto sceneNodes = g_engine->newArray();
                for (size_t sni = 0; sni < scene.nodes.size(); sni++) {
                    g_engine->setPropertyIndex(sceneNodes, sni, g_engine->newNumber(scene.nodes[sni]));
                }
                g_engine->setProperty(jsScene, "nodes", sceneNodes);

                g_engine->setPropertyIndex(jsScenes, si, jsScene);
            }
            g_engine->setProperty(result, "scenes", jsScenes);
            g_engine->setProperty(result, "defaultScene", g_engine->newNumber(gltfData->defaultScene));

            if (g_verboseLogging) {
                std::cout << "[GLTF] Loaded " << gltfData->meshes.size() << " meshes, "
                          << gltfData->materials.size() << " materials, "
                          << gltfData->images.size() << " images" << std::endl;
            }

            return result;
        })
    );

    engine->setGlobalProperty("Mystral", mystralNamespace);

    // ========================================================================
    // Native helper for offscreen canvas getContext('2d')
    // Called by the JS closure created in createElement('canvas')
    // ========================================================================
    engine->setGlobalProperty("__nativeGetContext2D",
        engine->newFunction("__nativeGetContext2D", [](void* ctx, const std::vector<js::JSValueHandle>& args) {
            if (args.size() < 2) {
                std::cerr << "[Canvas] __nativeGetContext2D requires contextType and canvasId" << std::endl;
                return g_engine->newNull();
            }

            std::string contextType = g_engine->toString(args[0]);
            int canvasId = static_cast<int>(g_engine->toNumber(args[1]));

            if (contextType != "2d") {
                std::cerr << "[Canvas] Unsupported context type for offscreen canvas: " << contextType << std::endl;
                return g_engine->newNull();
            }

            auto it = g_offscreenCanvases.find(canvasId);
            if (it == g_offscreenCanvases.end()) {
                std::cerr << "[Canvas] Canvas not found: " << canvasId << std::endl;
                return g_engine->newNull();
            }

            OffscreenCanvas* canvas = it->second.get();

            // Return cached context if already created
            if (canvas->hasContext2d) {
                return canvas->context2d;
            }

            // Get current dimensions from the canvas element (in case they were changed)
            std::string globalName = "__offscreenCanvas_" + std::to_string(canvasId);
            auto canvasElement = g_engine->getGlobalProperty(globalName.c_str());
            if (!g_engine->isNull(canvasElement) && !g_engine->isUndefined(canvasElement)) {
                auto widthProp = g_engine->getProperty(canvasElement, "width");
                auto heightProp = g_engine->getProperty(canvasElement, "height");
                if (!g_engine->isUndefined(widthProp)) {
                    canvas->width = static_cast<int>(g_engine->toNumber(widthProp));
                }
                if (!g_engine->isUndefined(heightProp)) {
                    canvas->height = static_cast<int>(g_engine->toNumber(heightProp));
                }
            }

            // Create Canvas 2D context with current dimensions
            if (g_verboseLogging) std::cout << "[Canvas] Creating offscreen 2D context (" << canvas->width << "x" << canvas->height << ")" << std::endl;
            canvas->context2d = canvas::createCanvas2DContext(g_engine, canvas->width, canvas->height);
            canvas->hasContext2d = true;
            g_engine->protect(canvas->context2d);
            return canvas->context2d;
        })
    );

    // ========================================================================
    // Global createOffscreenCanvas2D(width, height) helper
    // Creates an offscreen canvas with a 2D context at the specified size
    // This is easier to use than document.createElement('canvas').getContext('2d')
    // since it handles dimensions correctly
    // ========================================================================
    engine->setGlobalProperty("createOffscreenCanvas2D",
        engine->newFunction("createOffscreenCanvas2D", [](void* ctx, const std::vector<js::JSValueHandle>& args) {
            int width = 800;
            int height = 600;

            if (args.size() >= 1) {
                width = static_cast<int>(g_engine->toNumber(args[0]));
            }
            if (args.size() >= 2) {
                height = static_cast<int>(g_engine->toNumber(args[1]));
            }

            if (g_verboseLogging) std::cout << "[Canvas] Creating offscreen 2D canvas (" << width << "x" << height << ")" << std::endl;

            // Create a wrapper object that mimics a canvas with a 2D context
            auto canvasWrapper = g_engine->newObject();
            g_engine->setProperty(canvasWrapper, "width", g_engine->newNumber(width));
            g_engine->setProperty(canvasWrapper, "height", g_engine->newNumber(height));

            // Create the 2D context
            auto ctx2d = canvas::createCanvas2DContext(g_engine, width, height);
            g_engine->setProperty(canvasWrapper, "_context", ctx2d);

            // getContext('2d') returns the pre-created context
            g_engine->setProperty(canvasWrapper, "getContext",
                g_engine->newFunction("getContext", [](void* c, const std::vector<js::JSValueHandle>& a) {
                    // Get the stored context from the global (we need a way to access it)
                    // For now, return null and let callers use the _context directly
                    return g_engine->newNull();
                })
            );

            return canvasWrapper;
        })
    );

    if (g_verboseLogging) {
        std::cout << "[WebGPU] JavaScript bindings initialized" << std::endl;
        std::cout << "[WebGPU] createImageBitmap() available for image decoding" << std::endl;
    }

    return true;
#else
    std::cerr << "[WebGPU] No WebGPU backend available" << std::endl;
    return true;
#endif
}

// Getter for current texture (used by screenshot)
void* getCurrentRenderedTexture() {
#if defined(MYSTRAL_WEBGPU_WGPU) || defined(MYSTRAL_WEBGPU_DAWN)
    return g_currentTexture;
#else
    return nullptr;
#endif
}

uint32_t getCurrentTextureWidth() {
    return g_canvasWidth;
}

uint32_t getCurrentTextureHeight() {
    return g_canvasHeight;
}

void* getCurrentSurfaceTexture() {
#if defined(MYSTRAL_WEBGPU_WGPU) || defined(MYSTRAL_WEBGPU_DAWN)
    // Return the texture that the current view was created from (for screenshots)
    // or the current texture if no view was created yet
    return g_currentViewSourceTexture ? g_currentViewSourceTexture : g_currentTexture;
#else
    return nullptr;
#endif
}

// Screenshot buffer access
void* getScreenshotBuffer() {
#if defined(MYSTRAL_WEBGPU_WGPU) || defined(MYSTRAL_WEBGPU_DAWN)
    return g_screenshotBuffer;
#else
    return nullptr;
#endif
}

size_t getScreenshotBufferSize() {
    return g_screenshotBufferSize;
}

uint32_t getScreenshotBytesPerRow() {
    return g_screenshotBytesPerRow;
}

bool isScreenshotReady() {
    return g_screenshotReady;
}

void clearScreenshotReady() {
    g_screenshotReady = false;
}

void setOffscreenTexture(void* texture, void* textureView) {
    g_offscreenTexture = (WGPUTexture)texture;
    g_offscreenTextureView = (WGPUTextureView)textureView;
    if (g_verboseLogging) std::cout << "[WebGPU] Offscreen texture set for headless rendering" << std::endl;
}

void beginDawnFrame() {
    expireCanvasTexture();
}

// Canvas 2D compositing resources
static WGPUTexture g_canvas2DTexture = nullptr;
static WGPURenderPipeline g_canvas2DPipeline = nullptr;
static WGPUBindGroup g_canvas2DBindGroup = nullptr;
static WGPUSampler g_canvas2DSampler = nullptr;
static uint32_t g_canvas2DTextureWidth = 0;
static uint32_t g_canvas2DTextureHeight = 0;

static void cleanupInternalResources() {
    if (g_screenshotBuffer) {
        wgpuBufferDestroy(g_screenshotBuffer);
        wgpuBufferRelease(g_screenshotBuffer);
        g_screenshotBuffer = nullptr;
    }
    if (g_canvas2DBindGroup) { wgpuBindGroupRelease(g_canvas2DBindGroup); g_canvas2DBindGroup = nullptr; }
    if (g_canvas2DPipeline) { wgpuRenderPipelineRelease(g_canvas2DPipeline); g_canvas2DPipeline = nullptr; }
    if (g_canvas2DSampler) { wgpuSamplerRelease(g_canvas2DSampler); g_canvas2DSampler = nullptr; }
    if (g_canvas2DTexture) {
        wgpuTextureDestroy(g_canvas2DTexture);
        wgpuTextureRelease(g_canvas2DTexture);
        g_canvas2DTexture = nullptr;
    }
    g_screenshotBufferSize = 0; g_screenshotBytesPerRow = 0;
    g_screenshotPending = false; g_screenshotReady = false;
    g_screenshotData.clear();
    g_canvas2DTextureWidth = 0; g_canvas2DTextureHeight = 0;
    // These aliases are borrowed from Context/Canvas2D and have no API ref here.
    g_offscreenTexture = nullptr; g_offscreenTextureView = nullptr; g_mainCanvas2DContext = nullptr;
}

void compositeCanvas2DToWebGPU() {
    if (!g_mainCanvas2DContext || !g_device || !g_queue || !g_surface) {
        return;
    }

    // Get Canvas 2D pixel data
    const uint8_t* pixelData = g_mainCanvas2DContext->getPixelData();
    size_t pixelDataSize = g_mainCanvas2DContext->getPixelDataSize();
    int width = g_mainCanvas2DContext->getWidth();
    int height = g_mainCanvas2DContext->getHeight();

    if (!pixelData || pixelDataSize == 0) {
        return;
    }

    // Create or resize texture if needed
    if (!g_canvas2DTexture || g_canvas2DTextureWidth != (uint32_t)width || g_canvas2DTextureHeight != (uint32_t)height) {
        if (g_canvas2DTexture) {
            wgpuTextureDestroy(g_canvas2DTexture);
            wgpuTextureRelease(g_canvas2DTexture);
        }
        if (g_canvas2DBindGroup) {
            wgpuBindGroupRelease(g_canvas2DBindGroup);
            g_canvas2DBindGroup = nullptr;
        }

        WGPUTextureDescriptor texDesc = {};
        texDesc.size = {(uint32_t)width, (uint32_t)height, 1};
        texDesc.mipLevelCount = 1;
        texDesc.sampleCount = 1;
        texDesc.dimension = WGPUTextureDimension_2D;
        texDesc.format = WGPUTextureFormat_RGBA8Unorm;
        texDesc.usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst;

        g_canvas2DTexture = wgpuDeviceCreateTexture(g_device, &texDesc);
        g_canvas2DTextureWidth = width;
        g_canvas2DTextureHeight = height;
    }

    // Upload pixel data to texture
    WGPUImageCopyTexture_Compat destTexture = {};
    destTexture.texture = g_canvas2DTexture;
    destTexture.mipLevel = 0;
    destTexture.origin = {0, 0, 0};
    destTexture.aspect = WGPUTextureAspect_All;

    WGPUTextureDataLayout_Compat dataLayout = {};
    dataLayout.offset = 0;
    dataLayout.bytesPerRow = width * 4;
    dataLayout.rowsPerImage = height;

    WGPUExtent3D writeSize = {(uint32_t)width, (uint32_t)height, 1};
    wgpuQueueWriteTexture(g_queue, &destTexture, pixelData, pixelDataSize, &dataLayout, &writeSize);

    // Create pipeline if needed
    if (!g_canvas2DPipeline) {
        // Simple fullscreen quad shader
        const char* shaderCode = R"(
            @group(0) @binding(0) var texSampler: sampler;
            @group(0) @binding(1) var tex: texture_2d<f32>;

            struct VertexOutput {
                @builtin(position) position: vec4f,
                @location(0) uv: vec2f,
            }

            @vertex
            fn vs_main(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
                var positions = array<vec2f, 6>(
                    vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
                    vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0)
                );
                var uvs = array<vec2f, 6>(
                    vec2f(0.0, 1.0), vec2f(1.0, 1.0), vec2f(0.0, 0.0),
                    vec2f(0.0, 0.0), vec2f(1.0, 1.0), vec2f(1.0, 0.0)
                );
                var output: VertexOutput;
                output.position = vec4f(positions[vertexIndex], 0.0, 1.0);
                output.uv = uvs[vertexIndex];
                return output;
            }

            @fragment
            fn fs_main(input: VertexOutput) -> @location(0) vec4f {
                return textureSample(tex, texSampler, input.uv);
            }
        )";

        WGPUShaderModuleWGSLDescriptor_Compat wgslDesc = {};
        WGPUShaderModuleDescriptor shaderDesc = {};
        setupShaderModuleWGSL(&shaderDesc, &wgslDesc, shaderCode);

        WGPUShaderModule shaderModule = wgpuDeviceCreateShaderModule(g_device, &shaderDesc);

        // Create sampler
        WGPUSamplerDescriptor samplerDesc = {};
        samplerDesc.magFilter = WGPUFilterMode_Linear;
        samplerDesc.minFilter = WGPUFilterMode_Linear;
        samplerDesc.mipmapFilter = WGPUMipmapFilterMode_Linear;
        samplerDesc.addressModeU = WGPUAddressMode_ClampToEdge;
        samplerDesc.addressModeV = WGPUAddressMode_ClampToEdge;
        samplerDesc.addressModeW = WGPUAddressMode_ClampToEdge;
        samplerDesc.maxAnisotropy = 1;
        samplerDesc.lodMinClamp = 0.0f;
        samplerDesc.lodMaxClamp = 1.0f;
        g_canvas2DSampler = wgpuDeviceCreateSampler(g_device, &samplerDesc);

        // Create bind group layout
        WGPUBindGroupLayoutEntry bgLayoutEntries[2] = {};
        bgLayoutEntries[0].binding = 0;
        bgLayoutEntries[0].visibility = WGPUShaderStage_Fragment;
        bgLayoutEntries[0].sampler.type = WGPUSamplerBindingType_Filtering;
        bgLayoutEntries[1].binding = 1;
        bgLayoutEntries[1].visibility = WGPUShaderStage_Fragment;
        bgLayoutEntries[1].texture.sampleType = WGPUTextureSampleType_Float;
        bgLayoutEntries[1].texture.viewDimension = WGPUTextureViewDimension_2D;

        WGPUBindGroupLayoutDescriptor bgLayoutDesc = {};
        bgLayoutDesc.entryCount = 2;
        bgLayoutDesc.entries = bgLayoutEntries;
        WGPUBindGroupLayout bgLayout = wgpuDeviceCreateBindGroupLayout(g_device, &bgLayoutDesc);

        // Create pipeline layout
        WGPUPipelineLayoutDescriptor pipelineLayoutDesc = {};
        pipelineLayoutDesc.bindGroupLayoutCount = 1;
        pipelineLayoutDesc.bindGroupLayouts = &bgLayout;
        WGPUPipelineLayout pipelineLayout = wgpuDeviceCreatePipelineLayout(g_device, &pipelineLayoutDesc);

        // Create pipeline
        WGPURenderPipelineDescriptor pipelineDesc = {};
        pipelineDesc.layout = pipelineLayout;
        pipelineDesc.vertex.module = shaderModule;
        WGPU_SET_ENTRY_POINT(pipelineDesc.vertex, "vs_main");

        WGPUFragmentState fragmentState = {};
        fragmentState.module = shaderModule;
        WGPU_SET_ENTRY_POINT(fragmentState, "fs_main");
        fragmentState.targetCount = 1;

        WGPUColorTargetState colorTarget = {};
        colorTarget.format = g_surfaceFormat;
        colorTarget.writeMask = WGPUColorWriteMask_All;
        fragmentState.targets = &colorTarget;

        pipelineDesc.fragment = &fragmentState;
        pipelineDesc.primitive.topology = WGPUPrimitiveTopology_TriangleList;
        pipelineDesc.multisample.count = 1;
        pipelineDesc.multisample.mask = 0xFFFFFFFF;

        g_canvas2DPipeline = wgpuDeviceCreateRenderPipeline(g_device, &pipelineDesc);

        wgpuShaderModuleRelease(shaderModule);
        wgpuPipelineLayoutRelease(pipelineLayout);
        wgpuBindGroupLayoutRelease(bgLayout);

        if (!g_canvas2DPipeline) {
            std::cerr << "[Canvas2D] Failed to create compositing pipeline" << std::endl;
            return;
        }
    }

    // Create bind group (recreate if texture changed)
    if (!g_canvas2DBindGroup) {
        if (!g_canvas2DSampler || !g_canvas2DTexture) {
            return;
        }

        WGPUTextureViewDescriptor viewDesc = {};
        viewDesc.format = WGPUTextureFormat_RGBA8Unorm;
        viewDesc.dimension = WGPUTextureViewDimension_2D;
        viewDesc.baseMipLevel = 0;
        viewDesc.mipLevelCount = 1;
        viewDesc.baseArrayLayer = 0;
        viewDesc.arrayLayerCount = 1;
        WGPUTextureView texView = wgpuTextureCreateView(g_canvas2DTexture, &viewDesc);

        if (!texView) {
            return;
        }

        WGPUBindGroupEntry bgEntries[2] = {};
        bgEntries[0].binding = 0;
        bgEntries[0].sampler = g_canvas2DSampler;
        bgEntries[1].binding = 1;
        bgEntries[1].textureView = texView;

        WGPUBindGroupLayout layout = wgpuRenderPipelineGetBindGroupLayout(g_canvas2DPipeline, 0);
        if (!layout) {
            wgpuTextureViewRelease(texView);
            return;
        }

        WGPUBindGroupDescriptor bgDesc = {};
        bgDesc.layout = layout;
        bgDesc.entryCount = 2;
        bgDesc.entries = bgEntries;
        g_canvas2DBindGroup = wgpuDeviceCreateBindGroup(g_device, &bgDesc);

        wgpuBindGroupLayoutRelease(layout);
        wgpuTextureViewRelease(texView);

        if (!g_canvas2DBindGroup) {
            return;
        }
    }

    // Get surface texture
    WGPUSurfaceTexture surfaceTexture;
    wgpuSurfaceGetCurrentTexture(g_surface, &surfaceTexture);
    if (!wgpuSurfaceTextureStatusIsSuccess(surfaceTexture.status)) {
        return;
    }

    WGPUTextureViewDescriptor surfaceViewDesc = {};
    surfaceViewDesc.format = g_surfaceFormat;
    surfaceViewDesc.dimension = WGPUTextureViewDimension_2D;
    surfaceViewDesc.baseMipLevel = 0;
    surfaceViewDesc.mipLevelCount = 1;
    surfaceViewDesc.baseArrayLayer = 0;
    surfaceViewDesc.arrayLayerCount = 1;
    WGPUTextureView surfaceView = wgpuTextureCreateView(surfaceTexture.texture, &surfaceViewDesc);

    // Create command encoder and render pass
    WGPUCommandEncoderDescriptor encDesc = {};
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(g_device, &encDesc);

    WGPURenderPassColorAttachment colorAttachment = {};
    colorAttachment.view = surfaceView;
    colorAttachment.loadOp = WGPULoadOp_Clear;
    colorAttachment.storeOp = WGPUStoreOp_Store;
    colorAttachment.clearValue = {0.0, 0.0, 0.0, 1.0};
#if defined(MYSTRAL_WEBGPU_DAWN)
    colorAttachment.depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;
#endif

    WGPURenderPassDescriptor renderPassDesc = {};
    renderPassDesc.colorAttachmentCount = 1;
    renderPassDesc.colorAttachments = &colorAttachment;

    WGPURenderPassEncoder renderPass = wgpuCommandEncoderBeginRenderPass(encoder, &renderPassDesc);
    wgpuRenderPassEncoderSetPipeline(renderPass, g_canvas2DPipeline);
    wgpuRenderPassEncoderSetBindGroup(renderPass, 0, g_canvas2DBindGroup, 0, nullptr);
    wgpuRenderPassEncoderDraw(renderPass, 6, 1, 0, 0);
    wgpuRenderPassEncoderEnd(renderPass);
    wgpuRenderPassEncoderRelease(renderPass);

    // Copy rendered texture to screenshot buffer
    uint32_t bytesPerRow = ((g_canvasWidth * 4 + 255) / 256) * 256;  // Align to 256
    size_t requiredSize = bytesPerRow * g_canvasHeight;

    if (!g_screenshotBuffer || g_screenshotBufferSize < requiredSize) {
        if (g_screenshotBuffer) {
            wgpuBufferDestroy(g_screenshotBuffer);
            wgpuBufferRelease(g_screenshotBuffer);
        }

        WGPUBufferDescriptor bufferDesc = {};
        bufferDesc.size = requiredSize;
        bufferDesc.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead;
        bufferDesc.mappedAtCreation = false;

        g_screenshotBuffer = wgpuDeviceCreateBuffer(g_device, &bufferDesc);
        g_screenshotBufferSize = requiredSize;
        g_screenshotBytesPerRow = bytesPerRow;
    }

    // Copy surface texture to screenshot buffer
    WGPUImageCopyTexture_Compat srcCopy = {};
    srcCopy.texture = surfaceTexture.texture;
    srcCopy.mipLevel = 0;
    srcCopy.origin = {0, 0, 0};
    srcCopy.aspect = WGPUTextureAspect_All;

    WGPUImageCopyBuffer_Compat dstCopy = {};
    dstCopy.buffer = g_screenshotBuffer;
    dstCopy.layout.offset = 0;
    dstCopy.layout.bytesPerRow = bytesPerRow;
    dstCopy.layout.rowsPerImage = g_canvasHeight;

    WGPUExtent3D copySize = {g_canvasWidth, g_canvasHeight, 1};
    wgpuCommandEncoderCopyTextureToBuffer_Compat(encoder, &srcCopy, &dstCopy, &copySize);

    WGPUCommandBufferDescriptor cmdDesc = {};
    WGPUCommandBuffer cmdBuffer = wgpuCommandEncoderFinish(encoder, &cmdDesc);
    wgpuQueueSubmit(g_queue, 1, &cmdBuffer);

    wgpuCommandBufferRelease(cmdBuffer);
    wgpuCommandEncoderRelease(encoder);
    wgpuTextureViewRelease(surfaceView);

    // Present
    wgpuSurfacePresent(g_surface);

    // GetCurrentTexture returned an API reference independent of the Surface.
    // Screenshot data has already been copied above; keep no dangling alias.
    wgpuTextureRelease(surfaceTexture.texture);
    g_currentTexture = nullptr;
    g_screenshotReady = true;
}

// ============================================================================
// Video capture callback support (used by GPUReadbackRecorder)
// ============================================================================

// Video capture callback - called when queue.submit happens with a surface texture
// This allows the video recorder to capture frames without modifying the render loop
static void (*g_videoCaptureCallback)(void* texture, uint32_t width, uint32_t height, void* userData) = nullptr;
static void* g_videoCaptureUserData = nullptr;

void setVideoCaptureCallback(void (*callback)(void* texture, uint32_t width, uint32_t height, void* userData), void* userData) {
    g_videoCaptureCallback = callback;
    g_videoCaptureUserData = userData;
}

void clearVideoCaptureCallback() {
    g_videoCaptureCallback = nullptr;
    g_videoCaptureUserData = nullptr;
}

// Internal function to invoke video capture callback (called from queue.submit)
void invokeVideoCaptureCallback(WGPUTexture texture, uint32_t width, uint32_t height) {
    if (g_videoCaptureCallback && texture) {
        g_videoCaptureCallback(static_cast<void*>(texture), width, height, g_videoCaptureUserData);
    }
}

void endDawnFrame() {
    // Composite Canvas 2D content to WebGPU if the main canvas uses 2D context
    compositeCanvas2DToWebGPU();

    expireCanvasTexture();
    ++g_resourceFrame;
    logResourceSnapshot("periodic");

    // Tick the WebGPU device to process completed GPU work and free internal
    // resources (staging buffers, command encoder state, etc.). Without this,
    // internal objects accumulate unboundedly since completion callbacks never fire.
    if (g_device) {
#if defined(MYSTRAL_WEBGPU_DAWN)
        wgpuDeviceTick(g_device);
#elif defined(MYSTRAL_WEBGPU_WGPU)
        wgpuDevicePoll(g_device, false, nullptr);
#endif
    }
}

}  // namespace webgpu
}  // namespace mystral
