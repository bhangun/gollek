/**
 * platform/delegate_android.cpp
 * Delegate selection for Android (NNAPI, GPU via OpenCL/Vulkan, Hexagon DSP).
 *
 * Compiled only when __ANDROID__ is defined (via CMake target_compile_defs).
 *
 * Copyright (c) 2026 Kayys.tech — MIT License
 */

#include "gollek_engine_internal.h"

#ifdef __ANDROID__

#include "tensorflow/lite/delegates/gpu/delegate.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"

namespace gollek {

LitertDelegate *create_delegate(GollekDelegate preference,
                                std::string &error_out) {

  auto try_gpu = [&]() -> LitertDelegate * {
    LitertGpuDelegateOptionsV2 gpu_opts = LitertGpuDelegateOptionsV2Default();
    gpu_opts.inference_preference =
        TFLITE_GPU_INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER;
    gpu_opts.inference_priority1 = TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY;
    LitertDelegate *d = LitertGpuDelegateV2Create(&gpu_opts);
    if (!d)
      error_out = "Android GPU delegate creation failed";
    return d;
  };

  auto try_nnapi = [&]() -> LitertDelegate * {
    litert::StatefulNnApiDelegate::Options nnapi_opts;
    nnapi_opts.execution_preference =
        litert::StatefulNnApiDelegate::Options::kFastSingleAnswer;
    nnapi_opts.allow_fp16 = true;
    auto *d = new litert::StatefulNnApiDelegate(nnapi_opts);
    // StatefulNnApiDelegate inherits LitertDelegate
    return d;
  };

  switch (preference) {
  case GOLLEK_DELEGATE_GPU:
    return try_gpu();
  case GOLLEK_DELEGATE_NNAPI:
    return try_nnapi();

  case GOLLEK_DELEGATE_AUTO: {
    // Prefer NNAPI (covers NPU/DSP/GPU via Android runtime),
    // fall back to GPU delegate, then CPU.
    LitertDelegate *d = try_nnapi();
    if (d) {
      error_out.clear();
      return d;
    }
    d = try_gpu();
    if (d) {
      error_out.clear();
      return d;
    }
    return nullptr;
  }

  case GOLLEK_DELEGATE_HEXAGON: {
    // Hexagon delegate requires the libhexagon_nn_skel.so from
    // Qualcomm SDK — only available on Snapdragon devices.
    // Link: tensorflow/lite/delegates/hexagon
    error_out = "Hexagon delegate: link tensorflow/lite/delegates/hexagon "
                "and add the Qualcomm skel library to jniLibs.";
    return nullptr;
  }

  default:
    return nullptr;
  }
}

} // namespace gollek

#endif // __ANDROID__
