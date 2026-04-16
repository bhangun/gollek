/**
 * platform/delegate_desktop.cpp
 * Delegate selection for Linux / Windows / macOS-desktop.
 *
 * Compiled when neither __ANDROID__ nor __APPLE__ is defined,
 * or when GOLLEK_PLATFORM_DESKTOP is explicitly set.
 *
 * Copyright (c) 2026 Kayys.tech — MIT License
 */

#include "gollek_engine_internal.h"

#if !defined(__ANDROID__) && !defined(__APPLE__)

// Desktop GPU delegate (OpenCL path) is optional; guard with a feature flag.
#ifdef GOLLEK_ENABLE_GPU_DELEGATE
#include "tensorflow/lite/delegates/gpu/delegate.h"
#endif

namespace gollek {

LitertDelegate *create_delegate(GollekDelegate preference,
                                std::string &error_out) {

  (void)preference;

#ifdef GOLLEK_ENABLE_GPU_DELEGATE
  if (preference == GOLLEK_DELEGATE_GPU || preference == GOLLEK_DELEGATE_AUTO) {
    LitertGpuDelegateOptionsV2 opts = LitertGpuDelegateOptionsV2Default();
    LitertDelegate *d = LitertGpuDelegateV2Create(&opts);
    if (d)
      return d;
    error_out = "Desktop GPU delegate unavailable (no OpenCL?)";
  }
#else
  // XNNPACK is linked statically into TFLite >= 2.7 and applied
  // automatically when no other delegate is added — nothing to do here.
  (void)error_out;
#endif

  return nullptr; // Fall back to CPU + XNNPACK
}

} // namespace gollek

#endif // !Android && !Apple
