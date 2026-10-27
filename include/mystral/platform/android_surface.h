#pragma once

#if defined(__ANDROID__)
#include <android/native_window.h>
#include <cstdint>

namespace mystral {
namespace platform {

// Java publishes only real Surface callbacks. Each snapshot owns a window reference;
// the SDL/JavaScript owner releases it after use. No GPU work runs on the UI thread.
struct AndroidSurfaceState {
    ANativeWindow* window = nullptr;
    int generation = 0;
    uint64_t revision = 0;
};

AndroidSurfaceState acquireAndroidSurfaceState();
void clearAndroidSurfaceState();

} // namespace platform
} // namespace mystral
#endif
