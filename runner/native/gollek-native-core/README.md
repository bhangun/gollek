# Gollek Native Core — C++ Inference Engine

Cross-platform TFLite/LiteRT inference engine in C++, with bridges for
**Android (JNI/Kotlin)**, **iOS (ObjC++/Swift)**, **Web (Wasm/TS)**, and
**Desktop JVM (JNI/Java)**.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       Platform Consumers                        │
│                                                                 │
│  Android/Kotlin      iOS/Swift      Browser/TS    JVM/Java      │
│  GollekNativeBridge  GollekEngine   GollekEngine  GollekEngine  │
│  (Kotlin + JNI)      (ObjC++ + SW)  (TypeScript)  (Java + JNI) │
└──────────┬──────────────────┬──────────────┬──────────┬─────────┘
           │ JNI              │ direct C++   │ Wasm ABI │ JNI
┌──────────▼──────────────────▼──────────────▼──────────▼─────────┐
│                      gollek_engine.h (C API)                    │
│                                                                 │
│  gollek_engine_create / destroy                                 │
│  gollek_load_model_from_file / buffer                           │
│  gollek_set_input / invoke / get_output / infer                 │
│  gollek_get_input_info / output_info                            │
└──────────────────────────────┬──────────────────────────────────┘
                               │ C++ class
┌──────────────────────────────▼──────────────────────────────────┐
│                    gollek::Engine (C++ core)                    │
│                                                                 │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │  MemoryPool │  │ LitertModel* │  │ LitertInterpreter*   │   │
│  │  (slab      │  │ (unique_ptr) │  │ (unique_ptr + mutex)  │   │
│  │   allocator)│  └──────────────┘  └──────────────────────┘   │
│  └─────────────┘                                                │
└──────────────────────────────┬──────────────────────────────────┘
                               │ TFLite C API
┌──────────────────────────────▼──────────────────────────────────┐
│               TensorFlow Lite / LiteRT C API                    │
│              (libtensorflowlite_c.so / .dylib / .dll)           │
└──────────────────────────────┬──────────────────────────────────┘
                               │ delegate
┌──────────────────────────────▼──────────────────────────────────┐
│                   Hardware Acceleration                         │
│                                                                 │
│   Android: NNAPI (ANE/DSP/GPU) → GPU delegate (OpenCL/Vulkan)   │
│   iOS/macOS: Core ML (ANE) → Metal GPU delegate                 │
│   Desktop: XNNPACK (CPU SIMD) → OpenCL (optional)              │
│   Web/Wasm: CPU (SIMD via Emscripten)                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## File Map

```
gollek-native-core/
│
├── include/
│   ├── gollek_engine.h           ← Public C API (include this in all bridges)
│   └── gollek_engine_internal.h  ← Internal C++ class (Engine + MemoryPool)
│
├── src/
│   └── gollek_engine.cpp         ← C++ Engine implementation + C API wrappers
│
├── platform/
│   ├── delegate_android.cpp      ← NNAPI + GPU delegates (compiled on Android)
│   ├── delegate_apple.mm         ← Metal + CoreML delegates (compiled on Apple)
│   ├── delegate_desktop.cpp      ← XNNPACK / OpenCL (compiled on Linux/Win/macOS)
│   │
│   ├── android/
│   │   ├── gollek_jni.cpp        ← JNI glue (Android & Desktop JVM)
│   │   └── GollekNativeBridge.kt ← Kotlin API (replaces LiteRTNativeBindings.java)
│   │
│   ├── ios/
│   │   ├── GollekEngine.h        ← ObjC++ public header
│   │   ├── GollekEngine.mm       ← ObjC++ implementation
│   │   └── GollekEngine+Swift.swift ← Swift async/await extensions
│   │
│   ├── web/
│   │   ├── gollek_wasm_bridge.cpp ← Emscripten embind bindings
│   │   └── gollek_engine.ts      ← TypeScript API wrapper
│   │
│   └── desktop/
│       ├── gollek_desktop_runner.cpp ← C++ smoke-test driver
│       └── GollekDesktopRunner.java  ← Java RunnerPlugin (replaces LiteRTRunnerPlugin)
│
├── cmake/
│   └── gollek.map                ← GNU linker symbol export map
│
├── tests/
│   └── smoke_test_main.cpp       ← CLI smoke-test entry point
│
└── CMakeLists.txt                ← Unified build for all platforms
```

---

## Building

### Prerequisites

| Platform  | Requirements |
|-----------|-------------|
| Android   | NDK r25+, CMake 3.22+ |
| iOS       | Xcode 15+, CMake 3.22+ |
| Web       | Emscripten 3.1+, CMake 3.22+ |
| Desktop   | GCC 12+ / Clang 14+ / MSVC 2022, CMake 3.22+ |

All platforms need the **TFLite C library**. Use your existing `download_litert.sh`
or set `-DGOLLEK_FETCH_TFLITE=ON` to build from source.

---

### Android

```bash
# 1. Download TFLite for Android
# See: https://www.tensorflow.org/lite/guide/android

# 2. Build for arm64-v8a (most modern Android devices)
cmake -B build-android \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-26 \
  -DGOLLEK_PLATFORM=android \
  -DTFLITE_ROOT=/path/to/litert-android-arm64

cmake --build build-android --config Release

# Output: build-android/libgollek_core.so
# Place in: android-app/app/src/main/jniLibs/arm64-v8a/
```

**Kotlin usage:**
```kotlin
// In your Activity / ViewModel
GollekEngine(GollekConfig(delegate = GollekDelegate.AUTO)).use { engine ->
    engine.loadModelFromAssets(context, "models/mobilenet_v2.litertlm")
    val input  = ByteBuffer.allocateDirect(inputBytes)
    val output = ByteBuffer.allocateDirect(outputBytes)
    // fill input...
    engine.infer(input, output)
}
```

---

### iOS

```bash
# 1. Download LiteRT XCFramework from Google or build via CocoaPods:
#    pod 'TensorFlowLiteC', '~> 2.16'

# 2. Build static library
cmake -B build-ios -G Xcode \
  -DCMAKE_TOOLCHAIN_FILE=cmake/ios.toolchain.cmake \
  -DPLATFORM=OS64 \
  -DGOLLEK_PLATFORM=ios \
  -DTFLITE_ROOT=/path/to/litert-ios

cmake --build build-ios --config Release

# Output: build-ios/lib/libgollek_core.a
# Add to Xcode: Build Phases → Link Binary with Libraries
```

**Swift usage:**
```swift
import Foundation

let engine = try GollekEngineObjC(
    modelName: "mobilenet_v2",      // loads from main bundle
    config: .default()
)

let inputData: Data = buildInputTensor()
let output = try await engine.inferAsync(input: inputData)
let scores = output.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
```

---

### Web / WebAssembly

```bash
# 1. Install Emscripten
source /path/to/emsdk/emsdk_env.sh

# 2. Build TFLite for Wasm (or use a pre-built build)
# See: https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/pip_package

# 3. Build Gollek
emcmake cmake -B build-wasm \
  -DGOLLEK_PLATFORM=web \
  -DTFLITE_ROOT=/path/to/litert-wasm

cmake --build build-wasm

# Output: build-wasm/gollek_core.js + gollek_core.wasm
```

**TypeScript usage:**
```typescript
import { GollekEngine } from './gollek_engine';

const engine = await GollekEngine.create('./gollek_core.js', {
    numThreads: 4,
    delegate: Delegate.CPU,
});

await engine.loadModelFromUrl('/models/mobilenet_v2.litertlm');

// From a <canvas> element:
const tensor = GollekEngine.canvasToTensor(canvas);
const scores = engine.inferFloat32(tensor);

const topClass = scores.indexOf(Math.max(...scores));
console.log('Top class:', topClass, 'confidence:', scores[topClass]);

engine.destroy();
```

---

### Desktop / Server JVM

```bash
# 1. Build the shared library
cmake -B build-desktop \
  -DCMAKE_BUILD_TYPE=Release \
  -DGOLLEK_PLATFORM=desktop \
  -DTFLITE_ROOT=/path/to/litert-linux-x64

cmake --build build-desktop

# Output: build-desktop/libgollek_core.so (or .dylib / .dll)

# 2. Run smoke test
./build-desktop/gollek_test models/mobilenet_v1_1.0_224_quant.litertlm
```

**Java usage:**
```java
try (GollekEngine engine = new GollekEngine()) {
    engine.loadModelFromFile("/models/mobilenet_v2.litertlm");

    TensorMeta inMeta  = engine.inputMeta(0);
    TensorMeta outMeta = engine.outputMeta(0);

    ByteBuffer input  = ByteBuffer.allocateDirect((int) inMeta.byteSize());
    ByteBuffer output = ByteBuffer.allocateDirect((int) outMeta.byteSize());

    // fill input...
    engine.infer(input, output);
    // read output...
}
```

**Register as RunnerPlugin (Gollek server):**
```yaml
# application.yaml
inference:
  runners:
    - id: gollek-litert-runner
      enabled: true
      num-threads: 4
      delegate: AUTO
```

---

## Migration from Java FFM (LiteRTNativeBindings.java)

| Before (FFM)                          | After (C++ Core)                        |
|---------------------------------------|-----------------------------------------|
| `LiteRTNativeBindings.java` (~450 ln) | `gollek_jni.cpp` (JNI glue, ~200 ln)   |
| `LiteRTCpuRunner.java`                | `GollekDesktopRunner.java` (SPI impl)  |
| `LiteRTDelegateManager.java`          | `delegate_*.cpp` (platform-native)     |
| `LiteRTMemoryPool.java`               | `MemoryPool` in C++ (zero-GC)          |
| Java 21 Arena + MemorySegment         | `std::unique_ptr` + RAII in C++        |
| FFM downcall handles                  | JNI `native` methods (all JVMs)        |
| Linux/macOS/Windows only              | + Android, iOS, Web                    |

The `LiteRTTensorUtils`, `LiteRTBatchingManager`, `LiteRTErrorHandler`,
and `LiteRTMonitoring` Java classes remain in place — they work at the
Java layer above the native bridge and do not need porting.

---

## Memory model

The C++ `MemoryPool` pre-allocates a contiguous slab (default 16 MB) and uses
bump-pointer allocation per inference call, then resets atomically — no heap
fragmentation between inferences. On the JVM side, `ByteBuffer.allocateDirect()`
is used for zero-copy data transfer across the JNI boundary.

---

## Thread safety

`gollek::Engine::invoke()` is protected by a `std::mutex`. Multiple Java
threads may share a single `GollekEngine` instance, but invocations are
serialized. For true parallel throughput, create one `GollekEngine` per thread
or use the existing `LiteRTBatchingManager`.

---

## License

MIT — Copyright (c) 2026 Kayys.tech
