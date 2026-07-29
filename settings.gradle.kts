rootProject.name = "gollek-engine"

// Include alkhawarizm as a composite build so gollek can depend on alkhawarizm projects during development
includeBuild("../alkhawarizm")  // enabled: include local alkhawarizm composite build as source-of-truth for model modules
fun includeOptionalProject(projectPath: String, vararg candidatePaths: String) {
    val projectDir = candidatePaths
        .map { file(it) }
        .firstOrNull { candidate ->
            candidate.resolve("build.gradle.kts").isFile || candidate.resolve("build.gradle").isFile
        }
        ?: return

    include(projectPath)
    project(":$projectPath").projectDir = projectDir
}

// Optional core projects — include if their directories exist to avoid hard failures on partial checkouts
//includeOptionalProject("gollek-utils", "gollek-utils")
includeOptionalProject("core:gollek-adapter", "core/gollek-adapter")
includeOptionalProject("core:gollek-core", "core/gollek-core")
includeOptionalProject("core:gollek-error-code", "core/gollek-error-code")
includeOptionalProject("core:gollek-ir", "core/gollek-ir")
includeOptionalProject("core:gollek-ir-schema", "core/gollek-ir-schema")
includeOptionalProject("core:gollek-model-database", "core/gollek-model-database")
includeOptionalProject("core:gollek-model-repository", "core/gollek-model-repository")
includeOptionalProject("core:gollek-model-repo-hf", "core/gollek-model-repo-hf")
includeOptionalProject("core:gollek-model-repo-kaggle", "core/gollek-model-repo-kaggle")
includeOptionalProject("core:gollek-model-repo-local", "core/gollek-model-repo-local")
includeOptionalProject("core:gollek-model-runner", "core/gollek-model-runner")
includeOptionalProject("core:gollek-multitenancy", "core/gollek-multitenancy")
includeOptionalProject("core:gollek-observability", "core/gollek-observability")
includeOptionalProject("core:gollek-runtime-config", "core/gollek-runtime-config")
includeOptionalProject("core:gollek-tensor", "core/gollek-tensor")
includeOptionalProject("core:gollek-tokenizer-core", "core/gollek-tokenizer-core")
includeOptionalProject("core:gollek-tool-core", "core/gollek-tool-core")
// Dynamically include model projects if they exist (avoid hard failure when some model dirs are missing)
val staticallyIncludedModelProjects = setOf<String>()

// Include statically listed ones only if their directories exist (also check alkhawarizm/models)
staticallyIncludedModelProjects.forEach { name ->
    val alkhawarizmAlt = "../alkhawarizm/models/alkhawarizm-model-${name.removePrefix("gollek-model-")}"
    includeOptionalProject("models:$name", "models/$name", "../alkhawarizm/models/$name", alkhawarizmAlt)
}

// Auto-include any additional model projects present under any 'models' directories in the repository
val discoveredModelDirs = mutableSetOf<java.io.File>()
file(".").walkTopDown().maxDepth(5).forEach { f ->
    if (f.isDirectory && f.name == "models") {
        f.listFiles { candidate ->
            candidate.isDirectory &&
                    candidate.name.startsWith("gollek-model-") &&
                    candidate.name !in staticallyIncludedModelProjects &&
                    (candidate.resolve("build.gradle.kts").isFile || candidate.resolve("build.gradle").isFile)
        }?.forEach { discoveredModelDirs.add(it.absoluteFile) }
    }
}

discoveredModelDirs.sortedBy { it.name }.forEach { modelProject ->
    val logicalPath = "models:${modelProject.name}"
    includeOptionalProject(logicalPath, modelProject.absolutePath)
}
includeOptionalProject("suling", "../extensions/audio/suling", "stubs/suling")
includeOptionalProject("stubs:yaff-ffm-adapter", "stubs/yaff-ffm-adapter")

// Optional optimization plugins
includeOptionalProject("optimization:gollek-plugin-elastic-ep", "optimization/gollek-plugin-elastic-ep")
// includeOptionalProject("optimization:gollek-plugin-evicpress", "optimization/gollek-plugin-evicpress")
includeOptionalProject("optimization:gollek-plugin-fa3", "optimization/gollek-plugin-fa3")
includeOptionalProject("optimization:gollek-plugin-fa4", "optimization/gollek-plugin-fa4")
// includeOptionalProject("optimization:gollek-plugin-hybrid-attn", "optimization/gollek-plugin-hybrid-attn")
includeOptionalProject("optimization:gollek-plugin-kv-cache", "optimization/gollek-plugin-kv-cache")
includeOptionalProject("optimization:gollek-plugin-paged-attention", "optimization/gollek-plugin-paged-attention")
// includeOptionalProject("optimization:gollek-plugin-perfmode", "optimization/gollek-plugin-perfmode")
// includeOptionalProject("optimization:gollek-plugin-prefill-decode", "optimization/gollek-plugin-prefill-decode")
// includeOptionalProject("optimization:gollek-plugin-prompt-cache", "optimization/gollek-plugin-prompt-cache")
includeOptionalProject("optimization:gollek-plugin-qlora", "optimization/gollek-plugin-qlora")
// includeOptionalProject("optimization:gollek-plugin-wait-scheduler", "optimization/gollek-plugin-wait-scheduler")
// includeOptionalProject("optimization:gollek-plugin-weight-offload", "optimization/gollek-plugin-weight-offload")

// Optional plugins
includeOptionalProject("plugins:gollek-plugin-content-safety", "plugins/gollek-plugin-content-safety")
includeOptionalProject("plugins:gollek-plugin-mcp", "plugins/gollek-plugin-mcp")
includeOptionalProject("plugins:gollek-plugin-model-router", "plugins/gollek-plugin-model-router")
includeOptionalProject("plugins:wayang-plugin-openai", "../wayang/modules/provider/wayang-plugin-openai")
includeOptionalProject("plugins:wayang-plugin-anthropic", "../wayang/modules/provider/wayang-plugin-anthropic")
includeOptionalProject("plugins:wayang-plugin-gemini", "../wayang/modules/provider/wayang-plugin-gemini")
includeOptionalProject("plugins:wayang-plugin-cerebras", "../wayang/modules/provider/wayang-plugin-cerebras")
includeOptionalProject("plugins:wayang-plugin-mistral", "../wayang/modules/provider/wayang-plugin-mistral")
// includeOptionalProject("plugins:gollek-plugin-observability", "plugins/gollek-plugin-observability")
// includeOptionalProject("plugins:gollek-plugin-pii-redaction", "plugins/gollek-plugin-pii-redaction")
// includeOptionalProject("plugins:gollek-plugin-prompt", "plugins/gollek-plugin-prompt")
// includeOptionalProject("plugins:gollek-plugin-quota", "plugins/gollek-plugin-quota")
// includeOptionalProject("plugins:gollek-plugin-rag", "plugins/gollek-plugin-rag")
// includeOptionalProject("plugins:gollek-plugin-reasoning", "plugins/gollek-plugin-reasoning")
// includeOptionalProject("plugins:gollek-plugin-sampling", "plugins/gollek-plugin-sampling")
includeOptionalProject("plugins:gollek-plugin-semantic-cache", "plugins/gollek-plugin-semantic-cache")
// includeOptionalProject("plugins:gollek-plugin-streaming", "plugins/gollek-plugin-streaming")
// includeOptionalProject("plugins:gollek-safetensor-rag", "plugins/gollek-safetensor-rag")
includeOptionalProject("plugins:log-parser", "plugins/log-parser")
includeOptionalProject("plugins:gamelan", "plugins/gamelan")



// Optional quantizers
includeOptionalProject("quantizer:gollek-quantizer-autoround", "quantizer/gollek-quantizer-autoround")
includeOptionalProject("quantizer:gollek-quantizer-awq", "quantizer/gollek-quantizer-awq")
includeOptionalProject("quantizer:gollek-quantizer-gptq", "quantizer/gollek-quantizer-gptq")
includeOptionalProject("quantizer:gollek-quantizer-quip", "quantizer/gollek-quantizer-quip")
includeOptionalProject("quantizer:gollek-quantizer-turboquant", "quantizer/gollek-quantizer-turboquant")

// Optional runners and runtimes
includeOptionalProject("runner:gollek-diffusion", "runner/gollek-diffusion")
includeOptionalProject("runtime:gollek-runtime", "runtime/gollek-runtime")
includeOptionalProject("runtime:gollek-runtime-distributed", "runtime/gollek-runtime-distributed")

// Optional SDK
includeOptionalProject("sdk:gollek-sdk", "sdk/gollek-sdk")
includeOptionalProject("sdk:gollek-sdk-agent", "sdk/gollek-sdk-agent")
includeOptionalProject("sdk:gollek-sdk-api", "sdk/gollek-sdk-api")
includeOptionalProject("sdk:gollek-sdk-core", "sdk/gollek-sdk-core")
includeOptionalProject("sdk:gollek-sdk-local", "sdk/gollek-sdk-local")
includeOptionalProject("sdk:gollek-sdk-remote", "sdk/gollek-sdk-remote")
if (file("sdk/gollek-sdk-session").isDirectory) {
    include("sdk:gollek-sdk-session")
}

// Optional SPIs
includeOptionalProject("spi:gollek-spi", "spi/gollek-spi")
includeOptionalProject("spi:gollek-spi-inference", "spi/gollek-spi-inference")
includeOptionalProject("spi:gollek-spi-model", "spi/gollek-spi-model")
includeOptionalProject("spi:gollek-spi-multimodal", "spi/gollek-spi-multimodal")
includeOptionalProject("spi:gollek-spi-plugin", "spi/gollek-spi-plugin")
includeOptionalProject("spi:gollek-spi-runtime", "spi/gollek-spi-runtime")

// Optional UI
includeOptionalProject("ui:gollek-api", "ui/gollek-api")
includeOptionalProject("ui:gollek-cli", "ui/gollek-cli")

// Optional backends
includeOptionalProject("backend:blackwell:gollek-kernel-blackwell", "backend/blackwell/gollek-kernel-blackwell")
includeOptionalProject("backend:blackwell:gollek-plugin-kernel-blackwell", "backend/blackwell/gollek-plugin-kernel-blackwell")
includeOptionalProject("backend:cpu:gollek-backend-cpu", "backend/cpu/gollek-backend-cpu")
includeOptionalProject("backend:cuda:gollek-backend-cuda", "backend/cuda/gollek-backend-cuda")
includeOptionalProject("backend:cuda:gollek-kernel-cuda", "backend/cuda/gollek-kernel-cuda")
includeOptionalProject("backend:cuda:gollek-plugin-kernel-cuda", "backend/cuda/gollek-plugin-kernel-cuda")
includeOptionalProject("backend:directml:gollek-plugin-kernel-directml", "backend/directml/gollek-plugin-kernel-directml")
includeOptionalProject("backend:metal:gollek-backend-metal", "backend/metal/gollek-backend-metal")
includeOptionalProject("backend:metal:gollek-mlx-binding", "backend/metal/gollek-mlx-binding")
includeOptionalProject("backend:rocm:gollek-kernel-rocm", "backend/rocm/gollek-kernel-rocm")
includeOptionalProject("backend:rocm:gollek-plugin-kernel-rocm", "backend/rocm/gollek-plugin-kernel-rocm")

// Optional core plugins
includeOptionalProject("core:plugin:gollek-plugin-core", "core/plugin/gollek-plugin-core")
includeOptionalProject("core:plugin:gollek-plugin-kernel-core", "core/plugin/gollek-plugin-kernel-core")
includeOptionalProject("core:plugin:gollek-plugin-optimization-core", "core/plugin/gollek-plugin-optimization-core")
includeOptionalProject("core:plugin:gollek-plugin-runner-core", "core/plugin/gollek-plugin-runner-core")
includeOptionalProject("core:plugin:gollek-plugin-runner-gguf", "core/plugin/gollek-plugin-runner-gguf")

// Optional runners
includeOptionalProject("runner:diffuser:gollek-diffuser", "runner/diffuser/gollek-diffuser")
includeOptionalProject("runner:gguf:gollek-gguf-converter", "runner/gguf/gollek-gguf-converter")
includeOptionalProject("runner:gguf:gollek-gguf-converter-java", "runner/gguf/gollek-gguf-converter-java")
includeOptionalProject("runner:gguf:gollek-gguf-core", "runner/gguf/gollek-gguf-core")
includeOptionalProject("runner:litert:gollek-litert-core", "runner/litert/gollek-litert-core")
includeOptionalProject("runner:litert:gollek-plugin-runner-litert", "runner/litert/gollek-plugin-runner-litert")
includeOptionalProject("runner:litert:gollek-runner-litert", "runner/litert/gollek-runner-litert")

includeOptionalProject("runner:onnx:gollek-plugin-runner-onnx", "runner/onnx/gollek-plugin-runner-onnx")
includeOptionalProject("runner:onnx:gollek-runner-onnx", "runner/onnx/gollek-runner-onnx")

includeOptionalProject("runner:safetensor:gollek-runner-safetensor", "runner/safetensor/gollek-runner-safetensor")
includeOptionalProject("runner:safetensor:gollek-runner-stable-diffusion", "runner/safetensor/gollek-runner-stable-diffusion")
includeOptionalProject("runner:safetensor:gollek-safetensor-api", "runner/safetensor/gollek-safetensor-api")
includeOptionalProject("runner:safetensor:gollek-safetensor-core", "runner/safetensor/gollek-safetensor-core")
includeOptionalProject("runner:safetensor:gollek-safetensor-engine", "runner/safetensor/gollek-safetensor-engine")
includeOptionalProject("runner:safetensor:gollek-safetensor-loader", "runner/safetensor/gollek-safetensor-loader")
includeOptionalProject("runner:safetensor:gollek-safetensor-quantization", "runner/safetensor/gollek-safetensor-quantization")
includeOptionalProject("runner:safetensor:gollek-safetensor-spi", "runner/safetensor/gollek-safetensor-spi")
includeOptionalProject("runner:tensorrt:gollek-runner-tensorrt", "runner/tensorrt/gollek-runner-tensorrt")
includeOptionalProject("runner:tensorrt:gollek-plugin-runner-tensorrt", "runner/tensorrt/gollek-plugin-runner-tensorrt")
includeOptionalProject("runner:torch:gollek-runner-libtorch", "runner/torch/gollek-runner-libtorch")
includeOptionalProject("engine", "engine")

include("core:gollek-model-repo-hf")
include("core:gollek-model-repo-kaggle")
include("core:gollek-model-repo-local")
include("core:gollek-model-database")
include("core:gollek-tokenizer-core")
