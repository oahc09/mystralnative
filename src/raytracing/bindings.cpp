/**
 * MystralNative Ray Tracing JavaScript Bindings
 *
 * Implements the mystralRT global object for JavaScript access to
 * hardware ray tracing capabilities.
 *
 * API matches src/raytracing/types.ts MystralRT interface.
 */

#include "bindings.h"
#include "rt_common.h"
#include "mystral/js/engine.h"
#include "mystral/webgpu/context.h"
#include <iostream>
#include <unordered_map>
#include <cstring>

namespace mystral {
namespace rt {

// Global backend instance
static std::unique_ptr<IRTBackend> g_rtBackend = nullptr;

// Handle tracking for JS object cleanup
static uint32_t g_nextGeometryId = 1;
static uint32_t g_nextBLASId = 1;
static uint32_t g_nextTLASId = 1;

static std::unordered_map<uint32_t, RTGeometryHandle> g_geometries;
static std::unordered_map<uint32_t, RTBLASHandle> g_blases;
static std::unordered_map<uint32_t, RTTLASHandle> g_tlases;

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Extract Float32Array data from a JS value.
 * @param engine JS engine
 * @param value JS value (should be Float32Array)
 * @param outCount Output: number of floats
 * @return Pointer to float data, or nullptr on error
 */
static const float* extractFloat32Array(js::Engine* engine, js::JSValueHandle value, size_t* outCount) {
    size_t byteSize = 0;
    void* data = engine->getArrayBufferData(value, &byteSize);
    if (!data || byteSize == 0) {
        return nullptr;
    }
    if (outCount) {
        *outCount = byteSize / sizeof(float);
    }
    return static_cast<const float*>(data);
}

/**
 * Extract Uint32Array data from a JS value.
 * @param engine JS engine
 * @param value JS value (should be Uint32Array)
 * @param outCount Output: number of uint32s
 * @return Pointer to uint32 data, or nullptr on error
 */
static const uint32_t* extractUint32Array(js::Engine* engine, js::JSValueHandle value, size_t* outCount) {
    size_t byteSize = 0;
    void* data = engine->getArrayBufferData(value, &byteSize);
    if (!data || byteSize == 0) {
        return nullptr;
    }
    if (outCount) {
        *outCount = byteSize / sizeof(uint32_t);
    }
    return static_cast<const uint32_t*>(data);
}

/**
 * Create a JS geometry wrapper object.
 */
static js::JSValueHandle createGeometryJS(js::Engine* engine, uint32_t id) {
    auto obj = engine->newObject();
    engine->setProperty(obj, "_type", engine->newString("geometry"));
    engine->setProperty(obj, "_id", engine->newNumber(static_cast<double>(id)));
    return obj;
}

/**
 * Create a JS BLAS wrapper object.
 */
static js::JSValueHandle createBLASJS(js::Engine* engine, uint32_t id) {
    auto obj = engine->newObject();
    engine->setProperty(obj, "_type", engine->newString("blas"));
    engine->setProperty(obj, "_id", engine->newNumber(static_cast<double>(id)));
    return obj;
}

/**
 * Create a JS TLAS wrapper object.
 */
static js::JSValueHandle createTLASJS(js::Engine* engine, uint32_t id) {
    auto obj = engine->newObject();
    engine->setProperty(obj, "_type", engine->newString("tlas"));
    engine->setProperty(obj, "_id", engine->newNumber(static_cast<double>(id)));
    return obj;
}

/**
 * Get geometry ID from JS object.
 */
static uint32_t getGeometryId(js::Engine* engine, js::JSValueHandle obj) {
    auto idVal = engine->getProperty(obj, "_id");
    if (engine->isUndefined(idVal)) return 0;
    return static_cast<uint32_t>(engine->toNumber(idVal));
}

/**
 * Get BLAS ID from JS object.
 */
static uint32_t getBLASId(js::Engine* engine, js::JSValueHandle obj) {
    auto idVal = engine->getProperty(obj, "_id");
    if (engine->isUndefined(idVal)) return 0;
    return static_cast<uint32_t>(engine->toNumber(idVal));
}

/**
 * Get TLAS ID from JS object.
 */
static uint32_t getTLASId(js::Engine* engine, js::JSValueHandle obj) {
    auto idVal = engine->getProperty(obj, "_id");
    if (engine->isUndefined(idVal)) return 0;
    return static_cast<uint32_t>(engine->toNumber(idVal));
}

// ============================================================================
// JS Binding Functions
// ============================================================================

static js::JSValueHandle js_isSupported(void* ctx, const std::vector<js::JSValueHandle>& args, js::Engine* engine) {
    if (!g_rtBackend) {
        return engine->newBoolean(false);
    }
    return engine->newBoolean(g_rtBackend->isSupported());
}

static js::JSValueHandle js_getBackend(void* ctx, const std::vector<js::JSValueHandle>& args, js::Engine* engine) {
    if (!g_rtBackend) {
        return engine->newString("none");
    }
    return engine->newString(g_rtBackend->getBackend());
}

static js::JSValueHandle js_createGeometry(void* ctx, const std::vector<js::JSValueHandle>& args, js::Engine* engine) {
    if (!g_rtBackend || !g_rtBackend->isSupported()) {
        std::cerr << "[MystralRT] createGeometry: Hardware ray tracing not available" << std::endl;
        return engine->newNull();
    }

    if (args.empty() || !engine->isObject(args[0])) {
        std::cerr << "[MystralRT] createGeometry: Expected options object" << std::endl;
        return engine->newNull();
    }

    auto options = args[0];

    // Extract vertices (required)
    auto verticesVal = engine->getProperty(options, "vertices");
    size_t vertexCount = 0;
    const float* vertices = extractFloat32Array(engine, verticesVal, &vertexCount);
    if (!vertices || vertexCount == 0) {
        std::cerr << "[MystralRT] createGeometry: Invalid or missing vertices" << std::endl;
        return engine->newNull();
    }

    // Extract optional parameters
    auto indicesVal = engine->getProperty(options, "indices");
    size_t indexCount = 0;
    const uint32_t* indices = nullptr;
    if (!engine->isUndefined(indicesVal)) {
        indices = extractUint32Array(engine, indicesVal, &indexCount);
    }

    auto strideVal = engine->getProperty(options, "vertexStride");
    size_t vertexStride = engine->isUndefined(strideVal) ? 12 : static_cast<size_t>(engine->toNumber(strideVal));

    auto offsetVal = engine->getProperty(options, "vertexOffset");
    size_t vertexOffset = engine->isUndefined(offsetVal) ? 0 : static_cast<size_t>(engine->toNumber(offsetVal));

    // Build geometry description
    RTGeometryDesc desc = {};
    desc.vertices = vertices;
    desc.vertexCount = vertexCount / 3;  // vec3 positions
    desc.vertexStride = vertexStride;
    desc.vertexOffset = vertexOffset;
    desc.indices = indices;
    desc.indexCount = indexCount;

    // Create geometry
    RTGeometryHandle handle = g_rtBackend->createGeometry(desc);
    if (!handle._handle) {
        return engine->newNull();
    }

    // Store and return JS wrapper
    uint32_t id = g_nextGeometryId++;
    handle._id = id;
    g_geometries[id] = handle;

    return createGeometryJS(engine, id);
}

static js::JSValueHandle js_createBLAS(void* ctx, const std::vector<js::JSValueHandle>& args, js::Engine* engine) {
    if (!g_rtBackend || !g_rtBackend->isSupported()) {
        std::cerr << "[MystralRT] createBLAS: Hardware ray tracing not available" << std::endl;
        return engine->newNull();
    }

    if (args.empty() || !engine->isArray(args[0])) {
        std::cerr << "[MystralRT] createBLAS: Expected array of geometries" << std::endl;
        return engine->newNull();
    }

    auto geometriesArr = args[0];

    // Get array length
    auto lengthVal = engine->getProperty(geometriesArr, "length");
    size_t count = static_cast<size_t>(engine->toNumber(lengthVal));

    if (count == 0) {
        std::cerr << "[MystralRT] createBLAS: Empty geometry array" << std::endl;
        return engine->newNull();
    }

    // Collect geometry handles
    std::vector<RTGeometryHandle> handles;
    handles.reserve(count);

    for (size_t i = 0; i < count; i++) {
        auto geomObj = engine->getPropertyIndex(geometriesArr, static_cast<uint32_t>(i));
        uint32_t geomId = getGeometryId(engine, geomObj);

        auto it = g_geometries.find(geomId);
        if (it == g_geometries.end()) {
            std::cerr << "[MystralRT] createBLAS: Invalid geometry at index " << i << std::endl;
            return engine->newNull();
        }
        handles.push_back(it->second);
    }

    // Create BLAS
    RTBLASHandle handle = g_rtBackend->createBLAS(handles.data(), handles.size());
    if (!handle._handle) {
        return engine->newNull();
    }

    // Store and return JS wrapper
    uint32_t id = g_nextBLASId++;
    handle._id = id;
    g_blases[id] = handle;

    return createBLASJS(engine, id);
}

static js::JSValueHandle js_createTLAS(void* ctx, const std::vector<js::JSValueHandle>& args, js::Engine* engine) {
    if (!g_rtBackend || !g_rtBackend->isSupported()) {
        std::cerr << "[MystralRT] createTLAS: Hardware ray tracing not available" << std::endl;
        return engine->newNull();
    }

    if (args.empty() || !engine->isArray(args[0])) {
        std::cerr << "[MystralRT] createTLAS: Expected array of instances" << std::endl;
        return engine->newNull();
    }

    auto instancesArr = args[0];

    // Get array length
    auto lengthVal = engine->getProperty(instancesArr, "length");
    size_t count = static_cast<size_t>(engine->toNumber(lengthVal));

    if (count == 0) {
        std::cerr << "[MystralRT] createTLAS: Empty instance array" << std::endl;
        return engine->newNull();
    }

    // Collect instance descriptions
    std::vector<RTTLASInstance> instances;
    instances.reserve(count);

    for (size_t i = 0; i < count; i++) {
        auto instObj = engine->getPropertyIndex(instancesArr, static_cast<uint32_t>(i));

        RTTLASInstance inst = {};

        // Get BLAS reference
        auto blasObj = engine->getProperty(instObj, "blas");
        uint32_t blasId = getBLASId(engine, blasObj);
        auto it = g_blases.find(blasId);
        if (it == g_blases.end()) {
            std::cerr << "[MystralRT] createTLAS: Invalid BLAS at instance " << i << std::endl;
            return engine->newNull();
        }
        inst.blas = it->second;

        // Get transform (4x4 matrix as Float32Array)
        auto transformVal = engine->getProperty(instObj, "transform");
        size_t transformCount = 0;
        const float* transformData = extractFloat32Array(engine, transformVal, &transformCount);
        if (transformData && transformCount >= 16) {
            std::memcpy(inst.transform, transformData, 16 * sizeof(float));
        } else {
            // Default to identity matrix
            inst.transform[0] = 1.0f; inst.transform[5] = 1.0f;
            inst.transform[10] = 1.0f; inst.transform[15] = 1.0f;
        }

        // Get optional instance ID
        auto instanceIdVal = engine->getProperty(instObj, "instanceId");
        inst.instanceId = engine->isUndefined(instanceIdVal) ? 0 : static_cast<uint32_t>(engine->toNumber(instanceIdVal));

        inst.mask = 0xFF;  // Default visibility
        inst.flags = 0;

        instances.push_back(inst);
    }

    // Create TLAS
    RTTLASHandle handle = g_rtBackend->createTLAS(instances.data(), instances.size());
    if (!handle._handle) {
        return engine->newNull();
    }

    // Store and return JS wrapper
    uint32_t id = g_nextTLASId++;
    handle._id = id;
    g_tlases[id] = handle;

    return createTLASJS(engine, id);
}

static js::JSValueHandle js_updateTLAS(void* ctx, const std::vector<js::JSValueHandle>& args, js::Engine* engine) {
    if (!g_rtBackend || !g_rtBackend->isSupported()) {
        std::cerr << "[MystralRT] updateTLAS: Hardware ray tracing not available" << std::endl;
        return engine->newUndefined();
    }

    if (args.size() < 2) {
        std::cerr << "[MystralRT] updateTLAS: Expected (tlas, instances)" << std::endl;
        return engine->newUndefined();
    }

    // Get TLAS
    uint32_t tlasId = getTLASId(engine, args[0]);
    auto tlasIt = g_tlases.find(tlasId);
    if (tlasIt == g_tlases.end()) {
        std::cerr << "[MystralRT] updateTLAS: Invalid TLAS" << std::endl;
        return engine->newUndefined();
    }

    if (!engine->isArray(args[1])) {
        std::cerr << "[MystralRT] updateTLAS: Expected array of instances" << std::endl;
        return engine->newUndefined();
    }

    auto instancesArr = args[1];
    auto lengthVal = engine->getProperty(instancesArr, "length");
    size_t count = static_cast<size_t>(engine->toNumber(lengthVal));

    // Build instance array (same as createTLAS)
    std::vector<RTTLASInstance> instances;
    instances.reserve(count);

    for (size_t i = 0; i < count; i++) {
        auto instObj = engine->getPropertyIndex(instancesArr, static_cast<uint32_t>(i));

        RTTLASInstance inst = {};

        auto blasObj = engine->getProperty(instObj, "blas");
        uint32_t blasId = getBLASId(engine, blasObj);
        auto it = g_blases.find(blasId);
        if (it == g_blases.end()) {
            std::cerr << "[MystralRT] updateTLAS: Invalid BLAS at instance " << i << std::endl;
            return engine->newUndefined();
        }
        inst.blas = it->second;

        auto transformVal = engine->getProperty(instObj, "transform");
        size_t transformCount = 0;
        const float* transformData = extractFloat32Array(engine, transformVal, &transformCount);
        if (transformData && transformCount >= 16) {
            std::memcpy(inst.transform, transformData, 16 * sizeof(float));
        } else {
            inst.transform[0] = 1.0f; inst.transform[5] = 1.0f;
            inst.transform[10] = 1.0f; inst.transform[15] = 1.0f;
        }

        auto instanceIdVal = engine->getProperty(instObj, "instanceId");
        inst.instanceId = engine->isUndefined(instanceIdVal) ? 0 : static_cast<uint32_t>(engine->toNumber(instanceIdVal));

        inst.mask = 0xFF;
        inst.flags = 0;

        instances.push_back(inst);
    }

    g_rtBackend->updateTLAS(tlasIt->second, instances.data(), instances.size());
    return engine->newUndefined();
}

static js::JSValueHandle js_traceRays(void* ctx, const std::vector<js::JSValueHandle>& args, js::Engine* engine) {
    if (!g_rtBackend || !g_rtBackend->isSupported()) {
        std::cerr << "[MystralRT] traceRays: Hardware ray tracing not available" << std::endl;
        return engine->newUndefined();
    }

    if (args.empty() || !engine->isObject(args[0])) {
        std::cerr << "[MystralRT] traceRays: Expected options object" << std::endl;
        return engine->newUndefined();
    }

    auto options = args[0];

    // Get TLAS
    auto tlasObj = engine->getProperty(options, "tlas");
    uint32_t tlasId = getTLASId(engine, tlasObj);
    auto tlasIt = g_tlases.find(tlasId);
    if (tlasIt == g_tlases.end()) {
        std::cerr << "[MystralRT] traceRays: Invalid TLAS" << std::endl;
        return engine->newUndefined();
    }

    // Get dimensions
    auto widthVal = engine->getProperty(options, "width");
    auto heightVal = engine->getProperty(options, "height");
    uint32_t width = static_cast<uint32_t>(engine->toNumber(widthVal));
    uint32_t height = static_cast<uint32_t>(engine->toNumber(heightVal));

    // Get output texture (WebGPU texture handle)
    auto outputTextureVal = engine->getProperty(options, "outputTexture");
    void* outputTexture = webgpu::resolveResourceData(engine->getPrivateData(outputTextureVal));
    if (!outputTexture) return engine->newUndefined();

    TraceRaysOptions traceOptions = {};
    traceOptions.tlas = tlasIt->second;
    traceOptions.width = width;
    traceOptions.height = height;
    traceOptions.outputTexture = outputTexture;

    // Optional uniforms
    auto uniformsVal = engine->getProperty(options, "uniforms");
    if (!engine->isUndefined(uniformsVal)) {
        size_t uniformsSize = 0;
        traceOptions.uniforms = engine->getArrayBufferData(uniformsVal, &uniformsSize);
        traceOptions.uniformsSize = uniformsSize;
    }

    g_rtBackend->traceRays(traceOptions);
    return engine->newUndefined();
}

static js::JSValueHandle js_destroyBLAS(void* ctx, const std::vector<js::JSValueHandle>& args, js::Engine* engine) {
    if (args.empty()) return engine->newUndefined();

    uint32_t blasId = getBLASId(engine, args[0]);
    auto it = g_blases.find(blasId);
    if (it != g_blases.end()) {
        if (g_rtBackend) {
            g_rtBackend->destroyBLAS(it->second);
        }
        g_blases.erase(it);
    }
    return engine->newUndefined();
}

static js::JSValueHandle js_destroyTLAS(void* ctx, const std::vector<js::JSValueHandle>& args, js::Engine* engine) {
    if (args.empty()) return engine->newUndefined();

    uint32_t tlasId = getTLASId(engine, args[0]);
    auto it = g_tlases.find(tlasId);
    if (it != g_tlases.end()) {
        if (g_rtBackend) {
            g_rtBackend->destroyTLAS(it->second);
        }
        g_tlases.erase(it);
    }
    return engine->newUndefined();
}

static js::JSValueHandle js_destroyGeometry(void* ctx, const std::vector<js::JSValueHandle>& args, js::Engine* engine) {
    if (args.empty()) return engine->newUndefined();

    uint32_t geomId = getGeometryId(engine, args[0]);
    auto it = g_geometries.find(geomId);
    if (it != g_geometries.end()) {
        if (g_rtBackend) {
            g_rtBackend->destroyGeometry(it->second);
        }
        g_geometries.erase(it);
    }
    return engine->newUndefined();
}

// ============================================================================
// Public API
// ============================================================================

bool initializeRTBindings(js::Engine* engine) {
    if (!engine) {
        std::cerr << "[MystralRT] initializeRTBindings: No JS engine provided" << std::endl;
        return false;
    }

    // Create RT backend
    g_rtBackend = createRTBackend();

    // Create mystralRT global object
    auto mystralRT = engine->newObject();

    // Register methods
    engine->setProperty(mystralRT, "isSupported",
        engine->newFunction("isSupported", [engine](void* ctx, const std::vector<js::JSValueHandle>& args) {
            return js_isSupported(ctx, args, engine);
        })
    );

    engine->setProperty(mystralRT, "getBackend",
        engine->newFunction("getBackend", [engine](void* ctx, const std::vector<js::JSValueHandle>& args) {
            return js_getBackend(ctx, args, engine);
        })
    );

    engine->setProperty(mystralRT, "createGeometry",
        engine->newFunction("createGeometry", [engine](void* ctx, const std::vector<js::JSValueHandle>& args) {
            return js_createGeometry(ctx, args, engine);
        })
    );

    engine->setProperty(mystralRT, "createBLAS",
        engine->newFunction("createBLAS", [engine](void* ctx, const std::vector<js::JSValueHandle>& args) {
            return js_createBLAS(ctx, args, engine);
        })
    );

    engine->setProperty(mystralRT, "createTLAS",
        engine->newFunction("createTLAS", [engine](void* ctx, const std::vector<js::JSValueHandle>& args) {
            return js_createTLAS(ctx, args, engine);
        })
    );

    engine->setProperty(mystralRT, "updateTLAS",
        engine->newFunction("updateTLAS", [engine](void* ctx, const std::vector<js::JSValueHandle>& args) {
            return js_updateTLAS(ctx, args, engine);
        })
    );

    engine->setProperty(mystralRT, "traceRays",
        engine->newFunction("traceRays", [engine](void* ctx, const std::vector<js::JSValueHandle>& args) {
            return js_traceRays(ctx, args, engine);
        })
    );

    engine->setProperty(mystralRT, "destroyBLAS",
        engine->newFunction("destroyBLAS", [engine](void* ctx, const std::vector<js::JSValueHandle>& args) {
            return js_destroyBLAS(ctx, args, engine);
        })
    );

    engine->setProperty(mystralRT, "destroyTLAS",
        engine->newFunction("destroyTLAS", [engine](void* ctx, const std::vector<js::JSValueHandle>& args) {
            return js_destroyTLAS(ctx, args, engine);
        })
    );

    engine->setProperty(mystralRT, "destroyGeometry",
        engine->newFunction("destroyGeometry", [engine](void* ctx, const std::vector<js::JSValueHandle>& args) {
            return js_destroyGeometry(ctx, args, engine);
        })
    );

    // Register mystralRT as global
    engine->setGlobalProperty("mystralRT", mystralRT);

    std::cout << "[MystralRT] Bindings initialized (backend: " << g_rtBackend->getBackend() << ")" << std::endl;
    return true;
}

void cleanupRTBindings() {
    // Destroy all tracked resources
    for (auto& [id, handle] : g_tlases) {
        if (g_rtBackend) {
            g_rtBackend->destroyTLAS(handle);
        }
    }
    g_tlases.clear();

    for (auto& [id, handle] : g_blases) {
        if (g_rtBackend) {
            g_rtBackend->destroyBLAS(handle);
        }
    }
    g_blases.clear();

    for (auto& [id, handle] : g_geometries) {
        if (g_rtBackend) {
            g_rtBackend->destroyGeometry(handle);
        }
    }
    g_geometries.clear();

    // Destroy backend
    g_rtBackend.reset();

    // Reset ID counters
    g_nextGeometryId = 1;
    g_nextBLASId = 1;
    g_nextTLASId = 1;

    std::cout << "[MystralRT] Bindings cleaned up" << std::endl;
}

}  // namespace rt
}  // namespace mystral
