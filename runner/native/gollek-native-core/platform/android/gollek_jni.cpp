/**
 * platform/android/gollek_jni.cpp
 * JNI Bridge — Android (Java / Kotlin) ↔ Gollek C++ Core
 *
 * This replaces the FFM (Project Panama) bindings in LiteRTNativeBindings.java.
 * Each static native method in GollekNativeBridge.kt maps to one function here.
 *
 * Package: tech.kayys.gollek.native
 *
 * Copyright (c) 2026 Kayys.tech — MIT License
 */

#include "gollek_engine.h"
#include <jni.h>

#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <android/log.h>
#include <cstring>
#include <vector>

#define LOG_TAG "GollekJNI"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

/* ── Helper: GollekConfig from Kotlin data class ─────────────────────── */

static GollekConfig config_from_jobject(JNIEnv *env, jobject jconfig) {
  GollekConfig cfg = {};
  cfg.num_threads = 4;
  cfg.delegate = GOLLEK_DELEGATE_AUTO;
  cfg.enable_xnnpack = 1;
  cfg.use_memory_pool = 1;
  cfg.pool_size_bytes = 0; // Use default (16 MB)

  if (!jconfig)
    return cfg;

  jclass cls = env->GetObjectClass(jconfig);
  jfieldID fThreads = env->GetFieldID(cls, "numThreads", "I");
  jfieldID fDelegate = env->GetFieldID(cls, "delegate", "I");
  jfieldID fXnn = env->GetFieldID(cls, "enableXnnpack", "Z");
  jfieldID fPool = env->GetFieldID(cls, "useMemoryPool", "Z");
  jfieldID fPoolSz = env->GetFieldID(cls, "poolSizeBytes", "J");

  if (fThreads)
    cfg.num_threads = env->GetIntField(jconfig, fThreads);
  if (fDelegate)
    cfg.delegate = (GollekDelegate)env->GetIntField(jconfig, fDelegate);
  if (fXnn)
    cfg.enable_xnnpack = env->GetBooleanField(jconfig, fXnn) ? 1 : 0;
  if (fPool)
    cfg.use_memory_pool = env->GetBooleanField(jconfig, fPool) ? 1 : 0;
  if (fPoolSz)
    cfg.pool_size_bytes = (size_t)env->GetLongField(jconfig, fPoolSz);

  env->DeleteLocalRef(cls);
  return cfg;
}

/* ═══════════════════════════════════════════════════════════════════════
 * JNI — Lifecycle
 * ═══════════════════════════════════════════════════════════════════════ */

extern "C" JNIEXPORT jlong JNICALL
Java_tech_kayys_gollek_native_GollekNativeBridge_nativeCreate(JNIEnv *env,
                                                              jclass /*cls*/,
                                                              jobject jconfig) {
  GollekConfig cfg = config_from_jobject(env, jconfig);
  GollekEngineHandle h = gollek_engine_create(&cfg);
  LOGI("nativeCreate -> handle %p, version=%s", h, gollek_version());
  return reinterpret_cast<jlong>(h);
}

extern "C" JNIEXPORT void JNICALL
Java_tech_kayys_gollek_native_GollekNativeBridge_nativeDestroy(JNIEnv * /*env*/,
                                                               jclass /*cls*/,
                                                               jlong handle) {
  gollek_engine_destroy(reinterpret_cast<GollekEngineHandle>(handle));
  LOGI("nativeDestroy");
}

/* ═══════════════════════════════════════════════════════════════════════
 * JNI — Model loading
 * ═══════════════════════════════════════════════════════════════════════ */

extern "C" JNIEXPORT jint JNICALL
Java_tech_kayys_gollek_native_GollekNativeBridge_nativeLoadModelFromFile(
    JNIEnv *env, jclass /*cls*/, jlong handle, jstring jpath) {
  const char *path = env->GetStringUTFChars(jpath, nullptr);
  GollekStatus s = gollek_load_model_from_file(
      reinterpret_cast<GollekEngineHandle>(handle), path);
  env->ReleaseStringUTFChars(jpath, path);
  if (s != GOLLEK_OK)
    LOGE("loadModelFromFile failed: %s", gollek_status_string(s));
  return static_cast<jint>(s);
}

extern "C" JNIEXPORT jint JNICALL
Java_tech_kayys_gollek_native_GollekNativeBridge_nativeLoadModelFromBuffer(
    JNIEnv *env, jclass /*cls*/, jlong handle, jobject jbuffer) {
  void *data = env->GetDirectBufferAddress(jbuffer);
  size_t size = static_cast<size_t>(env->GetDirectBufferCapacity(jbuffer));
  GollekStatus s = gollek_load_model_from_buffer(
      reinterpret_cast<GollekEngineHandle>(handle), data, size);
  if (s != GOLLEK_OK)
    LOGE("loadModelFromBuffer failed: %s", gollek_status_string(s));
  return static_cast<jint>(s);
}

/**
 * Load a .litertlm file from the Android AssetManager (APK assets folder).
 * Kotlin call: bridge.loadModelFromAsset(assetManager,
 * "models/mobilenet.litertlm")
 */
extern "C" JNIEXPORT jint JNICALL
Java_tech_kayys_gollek_native_GollekNativeBridge_nativeLoadModelFromAsset(
    JNIEnv *env, jclass /*cls*/, jlong handle, jobject jasset_mgr,
    jstring jasset_path) {

  AAssetManager *mgr = AAssetManager_fromJava(env, jasset_mgr);
  const char *path = env->GetStringUTFChars(jasset_path, nullptr);

  AAsset *asset = AAssetManager_open(mgr, path, AASSET_MODE_BUFFER);
  env->ReleaseStringUTFChars(jasset_path, path);

  if (!asset) {
    LOGE("loadModelFromAsset: asset not found");
    return static_cast<jint>(GOLLEK_ERROR_MODEL_LOAD);
  }

  const void *data = AAsset_getBuffer(asset);
  size_t size = static_cast<size_t>(AAsset_getLength(asset));

  GollekStatus s = gollek_load_model_from_buffer(
      reinterpret_cast<GollekEngineHandle>(handle), data, size);
  AAsset_close(asset);

  if (s != GOLLEK_OK)
    LOGE("loadModelFromAsset failed: %s", gollek_status_string(s));
  return static_cast<jint>(s);
}

/* ═══════════════════════════════════════════════════════════════════════
 * JNI — Tensor info
 * ═══════════════════════════════════════════════════════════════════════ */

extern "C" JNIEXPORT jint JNICALL
Java_tech_kayys_gollek_native_GollekNativeBridge_nativeGetInputCount(
    JNIEnv * /*env*/, jclass /*cls*/, jlong handle) {
  return gollek_get_input_count(reinterpret_cast<GollekEngineHandle>(handle));
}

extern "C" JNIEXPORT jint JNICALL
Java_tech_kayys_gollek_native_GollekNativeBridge_nativeGetOutputCount(
    JNIEnv * /*env*/, jclass /*cls*/, jlong handle) {
  return gollek_get_output_count(reinterpret_cast<GollekEngineHandle>(handle));
}

/* Returns a long[] = { type, num_dims, dim0..dimN, byte_size } */
extern "C" JNIEXPORT jlongArray JNICALL
Java_tech_kayys_gollek_native_GollekNativeBridge_nativeGetInputInfo(
    JNIEnv *env, jclass /*cls*/, jlong handle, jint index) {
  GollekTensorInfo info = {};
  GollekStatus s = gollek_get_input_info(
      reinterpret_cast<GollekEngineHandle>(handle), index, &info);
  if (s != GOLLEK_OK)
    return nullptr;

  jlong arr[2 + GOLLEK_MAX_DIMS + 1] = {};
  arr[0] = info.type;
  arr[1] = info.num_dims;
  for (int i = 0; i < info.num_dims; ++i)
    arr[2 + i] = info.dims[i];
  arr[2 + info.num_dims] = static_cast<jlong>(info.byte_size);

  jlongArray result = env->NewLongArray(2 + info.num_dims + 1);
  env->SetLongArrayRegion(result, 0, 2 + info.num_dims + 1, arr);
  return result;
}

/* ═══════════════════════════════════════════════════════════════════════
 * JNI — Inference
 * ═══════════════════════════════════════════════════════════════════════ */

extern "C" JNIEXPORT jint JNICALL
Java_tech_kayys_gollek_native_GollekNativeBridge_nativeSetInput(
    JNIEnv *env, jclass /*cls*/, jlong handle, jint index, jobject jbuffer,
    jint bytes) {
  void *data = env->GetDirectBufferAddress(jbuffer);
  return static_cast<jint>(
      gollek_set_input(reinterpret_cast<GollekEngineHandle>(handle), index,
                       data, static_cast<size_t>(bytes)));
}

extern "C" JNIEXPORT jint JNICALL
Java_tech_kayys_gollek_native_GollekNativeBridge_nativeInvoke(JNIEnv * /*env*/,
                                                              jclass /*cls*/,
                                                              jlong handle) {
  return static_cast<jint>(
      gollek_invoke(reinterpret_cast<GollekEngineHandle>(handle)));
}

extern "C" JNIEXPORT jint JNICALL
Java_tech_kayys_gollek_native_GollekNativeBridge_nativeGetOutput(
    JNIEnv *env, jclass /*cls*/, jlong handle, jint index, jobject jbuffer,
    jint bytes) {
  void *dst = env->GetDirectBufferAddress(jbuffer);
  return static_cast<jint>(
      gollek_get_output(reinterpret_cast<GollekEngineHandle>(handle), index,
                        dst, static_cast<size_t>(bytes)));
}

extern "C" JNIEXPORT jstring JNICALL
Java_tech_kayys_gollek_native_GollekNativeBridge_nativeLastError(JNIEnv *env,
                                                                 jclass /*cls*/,
                                                                 jlong handle) {
  const char *err =
      gollek_last_error(reinterpret_cast<GollekEngineHandle>(handle));
  return env->NewStringUTF(err ? err : "");
}

extern "C" JNIEXPORT jstring JNICALL
Java_tech_kayys_gollek_native_GollekNativeBridge_nativeVersion(JNIEnv *env,
                                                               jclass /*cls*/) {
  return env->NewStringUTF(gollek_version());
}
