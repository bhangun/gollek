/**
 * platform/web/gollek_wasm_bridge.cpp
 * WebAssembly bridge — compiles via Emscripten to gollek_core.wasm + gollek_core.js
 *
 * Build command (run from project root):
 *   emcmake cmake -B build-wasm -DGOLLEK_PLATFORM=web
 *   cmake --build build-wasm
 *
 * The output is two files:
 *   gollek_core.js    — ES module loader with the JS API
 *   gollek_core.wasm  — the compiled binary
 *
 * Copyright (c) 2026 Kayys.tech — MIT License
 */

#include "gollek_engine.h"
#include <emscripten/emscripten.h>
#include <emscripten/bind.h>
#include <emscripten/val.h>
#include <vector>
#include <cstring>

using namespace emscripten;

/* ═══════════════════════════════════════════════════════════════════════
 * JavaScript-friendly wrapper class
 *
 * Emscripten's embind cannot expose raw pointers / C-style arrays directly,
 * so we wrap the opaque handle in a thin C++ class that uses std::vector
 * and val (JS ArrayBuffer) for data passing.
 * ═══════════════════════════════════════════════════════════════════════ */

class GollekEngineWasm {
public:
    GollekEngineWasm(int numThreads, int delegate,
                     bool enableXnnpack, bool useMemoryPool)
    {
        GollekConfig cfg = {};
        cfg.num_threads     = numThreads;
        cfg.delegate        = static_cast<GollekDelegate>(delegate);
        cfg.enable_xnnpack  = enableXnnpack ? 1 : 0;
        cfg.use_memory_pool = useMemoryPool ? 1 : 0;
        cfg.pool_size_bytes = 0;
        engine_ = gollek_engine_create(&cfg);
    }

    ~GollekEngineWasm() {
        gollek_engine_destroy(engine_);
    }

    /* ── Model loading ─────────────────────────────────────────────── */

    /**
     * loadModelFromBuffer(uint8Array)
     * JS: await engine.loadModelFromBuffer(new Uint8Array(modelBytes));
     */
    int loadModelFromBuffer(val uint8Array) {
        // Copy JS Uint8Array → C++ heap buffer
        const size_t size = uint8Array["length"].as<size_t>();
        model_buffer_.resize(size);
        val memory = val::module_property("HEAPU8");
        // Write through Emscripten's heap
        val typed_array = val::global("Uint8Array").new_(
            val::module_property("HEAPU8")["buffer"],
            reinterpret_cast<uintptr_t>(model_buffer_.data()),
            size
        );
        typed_array.call<void>("set", uint8Array);
        return static_cast<int>(
            gollek_load_model_from_buffer(engine_, model_buffer_.data(), size));
    }

    /* ── Tensor info ───────────────────────────────────────────────── */

    int inputCount()  { return gollek_get_input_count(engine_);  }
    int outputCount() { return gollek_get_output_count(engine_); }

    val getInputInfo(int index) {
        GollekTensorInfo info = {};
        if (gollek_get_input_info(engine_, index, &info) != GOLLEK_OK)
            return val::null();
        return pack_tensor_info(info);
    }

    val getOutputInfo(int index) {
        GollekTensorInfo info = {};
        if (gollek_get_output_info(engine_, index, &info) != GOLLEK_OK)
            return val::null();
        return pack_tensor_info(info);
    }

    /* ── Inference ─────────────────────────────────────────────────── */

    /**
     * setInput(index, uint8Array)
     * JS: engine.setInput(0, new Float32Array(pixelData).buffer);
     */
    int setInput(int index, val uint8Array) {
        const size_t bytes = uint8Array["byteLength"].as<size_t>();
        input_buf_.resize(bytes);
        // Copy from JS typed array into C++ buffer
        val heap_view = val::global("Uint8Array").new_(
            val::module_property("HEAPU8")["buffer"],
            reinterpret_cast<uintptr_t>(input_buf_.data()),
            bytes
        );
        // Use Uint8Array view over the incoming typed array
        val as_uint8 = val::global("Uint8Array").new_(uint8Array["buffer"],
                            uint8Array["byteOffset"], bytes);
        heap_view.call<void>("set", as_uint8);
        return static_cast<int>(
            gollek_set_input(engine_, index, input_buf_.data(), bytes));
    }

    int invoke() {
        return static_cast<int>(gollek_invoke(engine_));
    }

    /**
     * Returns a JS Uint8Array view of the output tensor.
     * JS: const out = engine.getOutput(0);
     *     const scores = new Float32Array(out.buffer);
     */
    val getOutput(int index) {
        GollekTensorInfo info = {};
        if (gollek_get_output_info(engine_, index, &info) != GOLLEK_OK)
            return val::null();
        output_buf_.resize(info.byte_size);
        if (gollek_get_output(engine_, index,
                              output_buf_.data(), output_buf_.size()) != GOLLEK_OK)
            return val::null();
        // Return a Uint8Array backed by Emscripten's HEAP
        val heap_view = val::global("Uint8Array").new_(
            val::module_property("HEAPU8")["buffer"],
            reinterpret_cast<uintptr_t>(output_buf_.data()),
            output_buf_.size()
        );
        return heap_view.call<val>("slice", 0);  // detached copy → safe to use
    }

    /**
     * Single-call convenience for single-input / single-output models.
     * JS: const result = engine.infer(inputUint8Array);
     */
    val infer(val inputUint8) {
        if (setInput(0, inputUint8) != GOLLEK_OK) return val::null();
        if (invoke()                != GOLLEK_OK) return val::null();
        return getOutput(0);
    }

    /* ── Diagnostics ───────────────────────────────────────────────── */

    std::string lastError() { return gollek_last_error(engine_) ?: ""; }
    std::string version()   { return gollek_version(); }

private:
    GollekEngineHandle     engine_       = nullptr;
    std::vector<uint8_t>   model_buffer_;    // keeps the flatbuffer alive
    std::vector<uint8_t>   input_buf_;
    std::vector<uint8_t>   output_buf_;

    static val pack_tensor_info(const GollekTensorInfo& info) {
        val obj = val::object();
        obj.set("name",     std::string(info.name ? info.name : ""));
        obj.set("type",     static_cast<int>(info.type));
        obj.set("byteSize", static_cast<int>(info.byte_size));
        obj.set("scale",    info.scale);
        obj.set("zeroPoint",info.zero_point);
        val dims = val::array();
        for (int i = 0; i < info.num_dims; ++i)
            dims.call<void>("push", info.dims[i]);
        obj.set("dims", dims);
        return obj;
    }
};

/* ═══════════════════════════════════════════════════════════════════════
 * Emscripten bindings
 * ═══════════════════════════════════════════════════════════════════════ */

EMSCRIPTEN_BINDINGS(gollek_module) {
    class_<GollekEngineWasm>("GollekEngine")
        .constructor<int, int, bool, bool>()

        // Model
        .function("loadModelFromBuffer", &GollekEngineWasm::loadModelFromBuffer)

        // Info
        .function("inputCount",   &GollekEngineWasm::inputCount)
        .function("outputCount",  &GollekEngineWasm::outputCount)
        .function("getInputInfo", &GollekEngineWasm::getInputInfo)
        .function("getOutputInfo",&GollekEngineWasm::getOutputInfo)

        // Inference
        .function("setInput",  &GollekEngineWasm::setInput)
        .function("invoke",    &GollekEngineWasm::invoke)
        .function("getOutput", &GollekEngineWasm::getOutput)
        .function("infer",     &GollekEngineWasm::infer)

        // Diagnostics
        .function("lastError", &GollekEngineWasm::lastError)
        .function("version",   &GollekEngineWasm::version)
        ;
}
