/** Android SDL entry for the native runtime. Java provides immutable assets/config and OS capture. */
#ifdef __ANDROID__

#include "mystral/runtime.h"
#include "mystral/platform/android_surface.h"
#include <SDL3/SDL.h>
#include <android/log.h>
#include <android/native_window_jni.h>
#include <jni.h>
#include <cstdlib>
#include <fstream>
#include <string>
#include <mutex>
#include <unistd.h>

#ifndef MYSTRAL_V8_EXTERNAL_STARTUP_DATA
#define MYSTRAL_V8_EXTERNAL_STARTUP_DATA 1
#endif

#if defined(MYSTRAL_JS_V8)
namespace mystral { namespace js { const char* getV8Version(); } }
#endif

namespace mystral {
namespace platform {
namespace {
std::mutex androidSurfaceMutex;
AndroidSurfaceState androidSurface;
}

AndroidSurfaceState acquireAndroidSurfaceState() {
    std::lock_guard<std::mutex> lock(androidSurfaceMutex);
    auto state = androidSurface;
    if (state.window) ANativeWindow_acquire(state.window);
    return state;
}

void clearAndroidSurfaceState() {
    std::lock_guard<std::mutex> lock(androidSurfaceMutex);
    if (androidSurface.window) ANativeWindow_release(androidSurface.window);
    androidSurface.window = nullptr;
    ++androidSurface.revision;
}
} // namespace platform
} // namespace mystral

extern "C" JNIEXPORT void JNICALL
Java_com_mystral_engine_MystralActivity_nativeSurfaceChanged(JNIEnv* env, jclass, jobject surface, jint generation) {
    auto* window = surface ? ANativeWindow_fromSurface(env, surface) : nullptr;
    uint64_t revision;
    {
        std::lock_guard<std::mutex> lock(mystral::platform::androidSurfaceMutex);
        auto& state = mystral::platform::androidSurface;
        if (generation < state.generation) {
            if (window) ANativeWindow_release(window);
            return;
        }
        if (state.window) ANativeWindow_release(state.window);
        state.window = window;
        state.generation = generation;
        revision = ++state.revision;
    }
    __android_log_print(ANDROID_LOG_INFO, "MystralInterop",
        "SURFACE_QUEUED generation=%d revision=%llu available=%d window=%p pid=%d",
        generation, static_cast<unsigned long long>(revision), window ? 1 : 0, window, static_cast<int>(getpid()));
}

static void exitStage(const char* stage) {
    __android_log_print(ANDROID_LOG_INFO, "MystralInterop", "EXIT_STAGE %s pid=%d thread=%llu",
        stage, static_cast<int>(getpid()), static_cast<unsigned long long>(SDL_GetCurrentThreadID()));
}

static const char* compiledAbi() {
#if defined(__x86_64__)
    return "x86_64";
#elif defined(__aarch64__)
    return "arm64-v8a";
#else
    return "unsupported";
#endif
}

static const char* compiledV8StartupMode() {
#if MYSTRAL_V8_EXTERNAL_STARTUP_DATA
    return "external";
#else
    return "embedded";
#endif
}

static const char* actualV8Version() {
#if defined(MYSTRAL_JS_V8)
    return mystral::js::getV8Version();
#else
    return "";
#endif
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_mystral_engine_MystralActivity_nativeCompiledAbi(JNIEnv* env, jclass) {
    return env->NewStringUTF(compiledAbi());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_mystral_engine_MystralActivity_nativeV8StartupMode(JNIEnv* env, jclass) {
    return env->NewStringUTF(compiledV8StartupMode());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_mystral_engine_MystralActivity_nativeV8Version(JNIEnv* env, jclass) {
    return env->NewStringUTF(actualV8Version());
}

extern "C" JNIEXPORT void JNICALL
Java_com_mystral_engine_MystralActivity_nativeRequestQuit(JNIEnv*, jclass) {
    exitStage("jni-quit-enter");
    SDL_Event event{};
    event.type = SDL_EVENT_QUIT;
    const bool pushed = SDL_PushEvent(&event);
    __android_log_print(pushed ? ANDROID_LOG_INFO : ANDROID_LOG_ERROR, "MystralInterop",
        "EXIT_STAGE sdl-quit-pushed result=%d error=%s", pushed ? 1 : 0, pushed ? "" : SDL_GetError());
}

static void notifyJava(const char* name, const char* signature, const char* error = nullptr, int code = 0) {
    auto* env = static_cast<JNIEnv*>(SDL_GetAndroidJNIEnv());
    auto activity = static_cast<jobject>(SDL_GetAndroidActivity());
    if (!env || !activity) return;
    jclass type = env->GetObjectClass(activity);
    jmethodID method = env->GetStaticMethodID(type, name, signature);
    if (method) {
        if (error) {
            jstring message = env->NewStringUTF(error);
            env->CallStaticVoidMethod(type, method, code, message);
            env->DeleteLocalRef(message);
        } else {
            jstring abi = env->NewStringUTF(compiledAbi());
            jstring engine = env->NewStringUTF(mystral::getJSEngine());
            jstring backend = env->NewStringUTF(mystral::getWebGPUBackend());
            jstring startupMode = env->NewStringUTF(compiledV8StartupMode());
            jstring v8Version = env->NewStringUTF(actualV8Version());
            env->CallStaticVoidMethod(type, method, abi, engine, backend, startupMode, v8Version);
            env->DeleteLocalRef(abi); env->DeleteLocalRef(engine); env->DeleteLocalRef(backend);
            env->DeleteLocalRef(startupMode); env->DeleteLocalRef(v8Version);
        }
    }
    if (env->ExceptionCheck()) {
        env->ExceptionDescribe(); env->ExceptionClear();
        __android_log_print(ANDROID_LOG_ERROR, "MystralInterop", "Java native identity callback failed");
    }
    env->DeleteLocalRef(type);
    env->DeleteLocalRef(activity);
}

static int reportedOutcome() {
    exitStage("reported-outcome-enter");
    auto* env = static_cast<JNIEnv*>(SDL_GetAndroidJNIEnv());
    auto activity = static_cast<jobject>(SDL_GetAndroidActivity());
    if (!env || !activity) return 1;
    jclass type = env->GetObjectClass(activity);
    jmethodID method = env->GetStaticMethodID(type, "getReportedOutcome", "()I");
    int value = method ? env->CallStaticIntMethod(type, method) : 1;
    if (env->ExceptionCheck()) { env->ExceptionDescribe(); env->ExceptionClear(); value = 1; }
    env->DeleteLocalRef(type); env->DeleteLocalRef(activity);
    __android_log_print(ANDROID_LOG_INFO, "MystralInterop", "EXIT_STAGE reported-outcome-return value=%d", value);
    return value == 0 ? 0 : 1;
}

extern "C" __attribute__((visibility("default"))) int SDL_main(int argc, char* argv[]) {
    int result = 1;
    std::string failure;
    if (argc < 3 || !argv[1] || !argv[2] || argv[1][0] != '/') {
        failure = "Native entry requires a private absolute entry and JSON configuration";
    } else {
#if MYSTRAL_V8_EXTERNAL_STARTUP_DATA
        if (argc < 4 || !argv[3] || argv[3][0] != '/') {
            failure = "External V8 requires a private absolute snapshot for the actual loaded ABI";
        } else {
            std::ifstream snapshot(argv[3], std::ios::binary | std::ios::ate);
            if (!snapshot || snapshot.tellg() <= 0 || setenv("MYSTRAL_V8_SNAPSHOT", argv[3], 1) != 0)
                failure = "Cannot select the packaged V8 snapshot for the actual loaded ABI";
        }
#endif
        if (failure.empty()) {
            mystral::RuntimeConfig config;
            config.width = 320;
            config.height = 240;
            config.title = "Mystral AIR Interop";
            config.fullscreen = true;
            auto runtime = mystral::Runtime::create(config);
            if (!runtime) failure = "Native runtime creation failed";
            else {
                notifyJava("onNativeReady",
                    "(Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;)V");
                std::string setup = "globalThis.__MYSTRAL_INTEROP__ = " + std::string(argv[2]) + ";";
                if (!runtime->evalScript(setup, "<android-interop-config>")) failure = "Interop configuration failed";
                else if (!runtime->loadScript(argv[1])) failure = "Packaged native entry failed to load";
                else {
                    // Keep the native process alive until Node has retained all artifacts.
                    exitStage("runtime-run-enter");
                    runtime->run();
                    exitStage("runtime-run-return");
                    result = runtime->getExitCode() == 0 ? reportedOutcome() : runtime->getExitCode();
                }
                exitStage("runtime-reset-enter");
                runtime.reset();
                exitStage("runtime-reset-return");
            }
        }
    }
    __android_log_print(failure.empty() ? ANDROID_LOG_INFO : ANDROID_LOG_ERROR,
        "MystralInterop", "Native SDL_main ended code=%d error=%s", result, failure.c_str());
    exitStage("java-exit-notify-enter");
    notifyJava("onNativeFinished", "(ILjava/lang/String;)V", failure.c_str(), result);
    exitStage("java-exit-notify-return");
    mystral::platform::clearAndroidSurfaceState();
    return result;
}

#endif
