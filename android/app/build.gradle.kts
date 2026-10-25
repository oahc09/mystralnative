plugins {
    id("com.android.application")
}

val jsEngine = providers.gradleProperty("mystralJsEngine").orElse("quickjs").get()
val gpuBackend = providers.gradleProperty("mystralGpuBackend").orElse("wgpu").get()
require(jsEngine in setOf("quickjs", "v8")) { "mystralJsEngine must be quickjs or v8" }
require(gpuBackend in setOf("wgpu", "dawn")) { "mystralGpuBackend must be wgpu or dawn" }
val interopAbis = providers.gradleProperty("interopAbis").orElse("arm64-v8a,x86_64").get()
    .split(',').map { it.trim() }.filter { it.isNotEmpty() }
require(interopAbis.isNotEmpty() && interopAbis.all { it in setOf("arm64-v8a", "x86_64") })
val androidDeps = rootProject.file(
    providers.gradleProperty("mystralAndroidDeps").orElse("../third_party/android").get()
)
val stagedSnapshots = layout.buildDirectory.dir("generated/interopSnapshots")
val stageV8Snapshots = tasks.register<Sync>("stageV8Snapshots") {
    into(stagedSnapshots)
    if (jsEngine == "v8") {
        for (abi in interopAbis) {
            from(androidDeps.resolve("v8/$abi")) {
                include("snapshot_blob.bin")
                into("v8/$abi")
            }
        }
        doFirst {
            for (abi in interopAbis) require(androidDeps.resolve("v8/$abi/snapshot_blob.bin").isFile) {
                "Missing the V8 snapshot for $abi"
            }
        }
    }
}

android {
    namespace = "com.mystral.engine"
    compileSdk = 36
    buildToolsVersion = "36.0.0"
    ndkVersion = providers.gradleProperty("mystralNdkVersion").orElse("29.0.14206865").get()

    defaultConfig {
        applicationId = "com.mystral.engine"
        minSdk = if (gpuBackend == "dawn") 26 else 24
        targetSdk = 34
        versionCode = 1
        versionName = "0.1.0"

        ndk {
            // Target modern 64-bit architectures
            // arm64-v8a: Most Android devices (ARM64)
            // x86_64: Android emulator on Intel/AMD
            abiFilters.addAll(interopAbis)
        }

        externalNativeBuild {
            cmake {
                // CMake arguments for the Mystral native build
                arguments.addAll(listOf(
                    "-DANDROID=ON",
                    "-DANDROID_STL=c++_shared",
                    "-DMYSTRAL_ANDROID_DEPS_ROOT=${androidDeps.invariantSeparatorsPath}",
                    "-DMYSTRAL_USE_QUICKJS=${if (jsEngine == "quickjs") "ON" else "OFF"}",
                    "-DMYSTRAL_USE_V8=${if (jsEngine == "v8") "ON" else "OFF"}",
                    "-DMYSTRAL_USE_JSC=OFF",
                    "-DMYSTRAL_USE_WGPU=${if (gpuBackend == "wgpu") "ON" else "OFF"}",
                    "-DMYSTRAL_USE_DAWN=${if (gpuBackend == "dawn") "ON" else "OFF"}",
                    "-DMYSTRAL_REQUIRE_CANVAS2D=${if (jsEngine == "v8" && gpuBackend == "dawn") "ON" else "OFF"}",
                    "-DMYSTRAL_USE_SWC=OFF",
                    "-DMYSTRAL_USE_DRACO=OFF",
                    "-DMYSTRAL_USE_QUICHE=OFF",
                    "-DMYSTRAL_USE_RAYTRACING=OFF"
                ))
                targets.add("mystral-runtime")
            }
        }
    }

    buildTypes {
        debug {
            applicationIdSuffix = ".interop"
        }
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }

    externalNativeBuild {
        cmake {
            // Point to the main CMakeLists.txt (parent of android directory)
            path = file("../../CMakeLists.txt")
            version = "3.22.1"
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_1_8
        targetCompatibility = JavaVersion.VERSION_1_8
    }

    buildFeatures {
        prefab = true  // Enable prefab for SDL3 AAR if we use it
    }

    sourceSets.getByName("main") {
        providers.gradleProperty("interopAssetsDir").orNull?.let {
            assets.setSrcDirs(listOf(rootProject.file(it)))
        }
        assets.srcDir(stagedSnapshots)
    }
    packaging {
        jniLibs {
            useLegacyPackaging = true
            keepDebugSymbols += setOf("**/libv8android.so", "**/libSDL3.so")
        }
    }
}

tasks.named("preBuild").configure { dependsOn(stageV8Snapshots) }

dependencies {
    // SDL3 AAR for Java bindings (SDLActivity, etc.)
    implementation(files(androidDeps.resolve("sdl3/SDL3-3.2.8.aar")))
}
