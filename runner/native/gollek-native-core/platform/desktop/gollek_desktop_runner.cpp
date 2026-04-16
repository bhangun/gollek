/**
 * platform/desktop/gollek_desktop_runner.cpp
 * Desktop (Linux / macOS / Windows) — direct C++ consumer
 *
 * This file shows how to use the Gollek engine from a plain C++ application
 * and also exposes a thin JNI entry point for the desktop JVM path
 * (replaces the Java 21 FFM bindings on non-Android JVMs).
 *
 * Copyright (c) 2026 Kayys.tech — MIT License
 */

#include "gollek_engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ═══════════════════════════════════════════════════════════════════════
 * Desktop CLI test driver
 * ═══════════════════════════════════════════════════════════════════════ */

/**
 * Quick smoke-test:
 *   ./gollek_test models/mobilenet_v2.litertlm
 *
 * Feeds a zeroed input tensor and prints the first 10 output bytes.
 */
int gollek_run_smoke_test(const char *model_path) {
  printf("Gollek %s  —  Smoke Test\n", gollek_version());
  printf("Model: %s\n\n", model_path);

  /* 1. Create engine (auto-select best delegate) */
  GollekConfig cfg = {0};
  cfg.num_threads = 4;
  cfg.delegate = GOLLEK_DELEGATE_AUTO;
  cfg.enable_xnnpack = 1;
  cfg.use_memory_pool = 1;

  GollekEngineHandle engine = gollek_engine_create(&cfg);
  if (!engine) {
    fprintf(stderr, "gollek_engine_create failed\n");
    return 1;
  }

  /* 2. Load model */
  GollekStatus s = gollek_load_model_from_file(engine, model_path);
  if (s != GOLLEK_OK) {
    fprintf(stderr, "Load failed: %s — %s\n", gollek_status_string(s),
            gollek_last_error(engine));
    gollek_engine_destroy(engine);
    return 1;
  }
  printf("[OK] Model loaded\n");

  /* 3. Inspect tensors */
  int n_in = gollek_get_input_count(engine);
  int n_out = gollek_get_output_count(engine);
  printf("     Inputs: %d  Outputs: %d\n", n_in, n_out);

  GollekTensorInfo in_info = {0};
  gollek_get_input_info(engine, 0, &in_info);
  printf("     Input[0]  name=%s  bytes=%zu  dims=[", in_info.name,
         in_info.byte_size);
  for (int i = 0; i < in_info.num_dims; ++i)
    printf("%d%s", in_info.dims[i], i < in_info.num_dims - 1 ? "," : "");
  printf("]\n");

  /* 4. Allocate zeroed input buffer */
  unsigned char *input_data = (unsigned char *)calloc(1, in_info.byte_size);
  if (!input_data) {
    fprintf(stderr, "OOM\n");
    gollek_engine_destroy(engine);
    return 1;
  }

  /* 5. Infer */
  GollekTensorInfo out_info = {0};
  gollek_get_output_info(engine, 0, &out_info);
  unsigned char *output_data = (unsigned char *)calloc(1, out_info.byte_size);

  s = gollek_infer(engine, input_data, in_info.byte_size, output_data,
                   out_info.byte_size);
  if (s != GOLLEK_OK) {
    fprintf(stderr, "Infer failed: %s — %s\n", gollek_status_string(s),
            gollek_last_error(engine));
  } else {
    printf("[OK] Inference complete\n");
    printf("     Output[0] bytes=%zu  first 10:", out_info.byte_size);
    for (size_t i = 0; i < 10 && i < out_info.byte_size; ++i)
      printf(" %02x", output_data[i]);
    printf("\n");
  }

  /* 6. Cleanup */
  free(input_data);
  free(output_data);
  gollek_engine_destroy(engine);
  printf("[OK] Resources freed\n");
  return s == GOLLEK_OK ? 0 : 1;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Desktop JNI — for the Java server/desktop path
 *
 * On Android, gollek_jni.cpp (in platform/android) is used.
 * On desktop JVM (Linux/macOS/Windows), we can reuse the same JNI
 * signatures against libgollek_core.so/.dylib/.dll.
 *
 * The Java side (GollekNativeBridge.kt) calls System.loadLibrary("gollek_core")
 * which maps to:
 *   Linux:   libgollek_core.so
 *   macOS:   libgollek_core.dylib
 *   Windows: gollek_core.dll
 *
 * We simply include the Android JNI file — the JNI signatures are identical
 * except for the asset-manager function (which returns an error on desktop).
 * ═══════════════════════════════════════════════════════════════════════ */

#if !defined(__ANDROID__) && defined(GOLLEK_ENABLE_DESKTOP_JNI)
#include "../android/gollek_jni.cpp" // reuse identical JNI glue

// Override the Android-specific asset function with a desktop stub
extern "C" JNIEXPORT jint JNICALL
Java_tech_kayys_gollek_native_GollekNativeBridge_nativeLoadModelFromAsset(
    JNIEnv *env, jclass, jlong, jobject, jstring jasset_path) {
  // Desktop doesn't have an AssetManager; tell the caller.
  const char *path = env->GetStringUTFChars(jasset_path, nullptr);
  fprintf(stderr,
          "GollekJNI: AssetManager not available on desktop (%s)\n"
          "           Use loadModelFromFile instead.\n",
          path);
  env->ReleaseStringUTFChars(jasset_path, path);
  return static_cast<jint>(GOLLEK_ERROR_INVALID_ARG);
}
#endif // desktop JNI
