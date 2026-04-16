/**
 * platform/delegate_apple.mm
 * Delegate selection for iOS and macOS (Metal GPU, Core ML).
 *
 * This is an Objective-C++ file (.mm) so it can import Apple frameworks.
 * Compiled only when __APPLE__ is defined.
 *
 * Copyright (c) 2026 Kayys.tech — MIT License
 */

#include "gollek_engine_internal.h"

#ifdef __APPLE__

#include "tensorflow/lite/delegates/coreml/coreml_delegate.h"
#include "tensorflow/lite/delegates/gpu/metal_delegate.h"
#import <Foundation/Foundation.h>

namespace gollek {

LitertDelegate *create_delegate(GollekDelegate preference,
                                std::string &error_out) {

  auto try_metal = [&]() -> LitertDelegate * {
    TFLGpuDelegateOptions opts = TFLGpuDelegateOptionsDefault();
    opts.allow_precision_loss = false; // Keep FP32 accuracy
    opts.wait_type = TFLGpuDelegateWaitType::TFLGpuDelegateWaitTypePassive;
    LitertDelegate *d = TFLGpuDelegateCreate(&opts);
    if (!d)
      error_out = "Metal GPU delegate creation failed";
    return d;
  };

  auto try_coreml = [&]() -> LitertDelegate * {
    LitertCoreMlDelegateOptions cml_opts = {};
    cml_opts.enabled_devices = LitertCoreMlDelegateAllDevices;
    cml_opts.coreml_version = 3;           // Core ML 3+ supports ANE
    cml_opts.max_delegated_partitions = 0; // 0 = no limit
    LitertDelegate *d = LitertCoreMlDelegateCreate(&cml_opts);
    if (!d)
      error_out = "Core ML delegate creation failed";
    return d;
  };

  switch (preference) {
  case GOLLEK_DELEGATE_GPU:
    return try_metal();
  case GOLLEK_DELEGATE_COREML:
    return try_coreml();

  case GOLLEK_DELEGATE_AUTO: {
    // On Apple Silicon prefer Core ML (ANE), then Metal, then CPU.
    LitertDelegate *d = try_coreml();
    if (d) {
      error_out.clear();
      return d;
    }
    d = try_metal();
    if (d) {
      error_out.clear();
      return d;
    }
    return nullptr;
  }

  default:
    return nullptr;
  }
}

} // namespace gollek

#endif // __APPLE__
