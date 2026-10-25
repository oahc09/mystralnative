/** Android SDL entry for the native runtime. Java provides immutable assets/config and OS capture. */
#ifdef __ANDROID__

#include "mystral/runtime.h"
#include <SDL3/SDL.h>
#include <android/log.h>
#include <jni.h>
#include <cstdlib>
#include <fstream>
#include <string>
#include <unistd.h>

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

extern "C" JNIEXPORT jstring JNICALL
Java_com_mystral_engine_MystralActivity_nativeCompiledAbi(JNIEnv* env, jclass) {
    return env->NewStringUTF(compiledAbi());
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
            env->CallStaticVoidMethod(type, method, abi, engine, backend);
            env->DeleteLocalRef(abi); env->DeleteLocalRef(engine); env->DeleteLocalRef(backend);
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
    if (argc < 4 || !argv[1] || !argv[2] || !argv[3] || argv[1][0] != '/' || argv[3][0] != '/') {
        failure = "Native entry requires a private absolute entry, JSON configuration and V8 snapshot";
    } else {
        std::ifstream snapshot(argv[3], std::ios::binary | std::ios::ate);
        if (!snapshot || snapshot.tellg() <= 0 || setenv("MYSTRAL_V8_SNAPSHOT", argv[3], 1) != 0) {
            failure = "Cannot select the packaged V8 snapshot for the actual loaded ABI";
        } else {
            mystral::RuntimeConfig config;
            config.width = 320;
            config.height = 240;
            config.title = "Mystral AIR Interop";
            config.fullscreen = true;
            auto runtime = mystral::Runtime::create(config);
            if (!runtime) failure = "Native runtime creation failed";
            else {
                notifyJava("onNativeReady", "(Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;)V");
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
    return result;
}

#endif
