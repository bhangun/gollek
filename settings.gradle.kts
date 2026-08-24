rootProject.name = "gollek-engine"

// Include alkhawarizm as a composite build so gollek can depend on alkhawarizm projects during development
includeBuild("../alkhawarizm")  // enabled: include local alkhawarizm composite build as source-of-truth for model modules
includeBuild("../tafkir")       // enabled: include local tafkir composite build for quantizers
fun includeOptionalProject(projectPath: String, vararg candidatePaths: String) {
    val projectDir = candidatePaths
        .map { file(it) }
        .firstOrNull { candidate ->
            candidate.resolve("build.gradle.kts").isFile || candidate.resolve("build.gradle").isFile
        }
        ?: return

    val parts = projectPath.split(":")
    var currentPath = ""
    for (i in 0 until parts.size - 1) {
        val part = parts[i]
        currentPath = if (currentPath.isEmpty()) part else "$currentPath:$part"
        include(currentPath)
        
        val rootPart = parts[0]
        var mappedDir = currentPath.replace(":", "/")
        if (rootPart in listOf("core", "spi", "sdk", "plugin", "observability", "repository")) {
            mappedDir = "framework/" + mappedDir
        } else if (rootPart in listOf("backend", "runner", "quantizer", "optimization", "plugins", "repository", "adapter")) {
            mappedDir = "runtime/" + mappedDir
        } else if (rootPart == "runtime") {
            mappedDir = "runtime/core/" + mappedDir.removePrefix("runtime").removePrefix("/")
        }
        project(":$currentPath").projectDir = file(mappedDir)
    }

    include(projectPath)
    project(":$projectPath").projectDir = projectDir
}

// Optional core projects — include if their directories exist to avoid hard failures on partial checkouts
//includeOptionalProject("gollek-utils", "gollek-utils")
includeOptionalProject("core:gollek-adapter", "runtime/adapter/gollek-adapter")
includeOptionalProject("core:gollek-core", "framework/core/gollek-core")
includeOptionalProject("repository:gollek-model-database", "framework/repository/gollek-model-database")
includeOptionalProject("repository:gollek-model-repo-common", "runtime/repository/gollek-model-repo-common")
includeOptionalProject("core:gollek-model-repo-hf", "runtime/repository/gollek-model-repo-hf")
includeOptionalProject("core:gollek-model-repo-kaggle", "runtime/repository/gollek-model-repo-kaggle")
includeOptionalProject("core:gollek-model-repo-local", "runtime/repository/gollek-model-repo-local")
includeOptionalProject("spi:gollek-spi-runner", "framework/spi/gollek-spi-runner")
includeOptionalProject("observability:gollek-observability", "framework/observability/gollek-observability")
includeOptionalProject("core:gollek-tensor", "framework/core/gollek-tensor")
includeOptionalProject("core:gollek-tokenizer-core", "framework/core/gollek-tokenizer-core")
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

// Optional optimization plugins
// includeOptionalProject("optimization:gollek-plugin-elastic-ep", "runtime/optimization/gollek-plugin-elastic-ep")
// includeOptionalProject("optimization:gollek-plugin-evicpress", "runtime/optimization/gollek-plugin-evicpress")
// includeOptionalProject("optimization:gollek-plugin-fa3", "runtime/optimization/gollek-plugin-fa3")
// includeOptionalProject("optimization:gollek-plugin-fa4", "runtime/optimization/gollek-plugin-fa4")
// includeOptionalProject("optimization:gollek-plugin-hybrid-attn", "runtime/optimization/gollek-plugin-hybrid-attn")
includeOptionalProject("optimization:gollek-plugin-kv-cache", "runtime/optimization/gollek-plugin-kv-cache")
// includeOptionalProject("optimization:gollek-plugin-paged-attention", "runtime/optimization/gollek-plugin-paged-attention")
// includeOptionalProject("optimization:gollek-plugin-perfmode", "runtime/optimization/gollek-plugin-perfmode")
// includeOptionalProject("optimization:gollek-plugin-prefill-decode", "runtime/optimization/gollek-plugin-prefill-decode")
// includeOptionalProject("optimization:gollek-plugin-prompt-cache", "runtime/optimization/gollek-plugin-prompt-cache")
// includeOptionalProject("optimization:gollek-plugin-qlora", "runtime/optimization/gollek-plugin-qlora")
// includeOptionalProject("optimization:gollek-plugin-wait-scheduler", "runtime/optimization/gollek-plugin-wait-scheduler")
// includeOptionalProject("optimization:gollek-plugin-weight-offload", "runtime/optimization/gollek-plugin-weight-offload")

// Optional plugins
includeOptionalProject("plugins:gollek-plugin-content-safety", "runtime/plugins/gollek-plugin-content-safety")
includeOptionalProject("plugins:gollek-plugin-mcp", "runtime/plugins/gollek-plugin-mcp")
includeOptionalProject("plugins:gollek-plugin-model-router", "runtime/plugins/gollek-plugin-model-router")
includeOptionalProject("plugins:wayang-plugin-openai", "../wayang/modules/provider/wayang-plugin-openai")
includeOptionalProject("plugins:wayang-plugin-anthropic", "../wayang/modules/provider/wayang-plugin-anthropic")
includeOptionalProject("plugins:wayang-plugin-gemini", "../wayang/modules/provider/wayang-plugin-gemini")
includeOptionalProject("plugins:wayang-plugin-cerebras", "../wayang/modules/provider/wayang-plugin-cerebras")
includeOptionalProject("plugins:wayang-plugin-mistral", "../wayang/modules/provider/wayang-plugin-mistral")
// includeOptionalProject("plugins:gollek-plugin-observability", "runtime/plugins/gollek-plugin-observability")
// includeOptionalProject("plugins:gollek-plugin-pii-redaction", "runtime/plugins/gollek-plugin-pii-redaction")
// includeOptionalProject("plugins:gollek-plugin-prompt", "runtime/plugins/gollek-plugin-prompt")
// includeOptionalProject("plugins:gollek-plugin-quota", "runtime/plugins/gollek-plugin-quota")
// includeOptionalProject("plugins:gollek-plugin-rag", "runtime/plugins/gollek-plugin-rag")
// includeOptionalProject("plugins:gollek-plugin-reasoning", "runtime/plugins/gollek-plugin-reasoning")
// includeOptionalProject("plugins:gollek-plugin-sampling", "runtime/plugins/gollek-plugin-sampling")
includeOptionalProject("plugins:gollek-plugin-semantic-cache", "runtime/plugins/gollek-plugin-semantic-cache")
// includeOptionalProject("plugins:gollek-plugin-streaming", "runtime/plugins/gollek-plugin-streaming")
// includeOptionalProject("plugins:gollek-safetensor-rag", "runtime/plugins/gollek-safetensor-rag")
includeOptionalProject("plugins:log-parser", "runtime/plugins/log-parser")
includeOptionalProject("plugins:gamelan", "runtime/plugins/gamelan")



// Optional quantizers
includeOptionalProject("quantizer:gollek-quantizer-autoround", "runtime/quantizer/gollek-quantizer-autoround")
includeOptionalProject("quantizer:gollek-quantizer-awq", "runtime/quantizer/gollek-quantizer-awq")
includeOptionalProject("quantizer:gollek-quantizer-gptq", "runtime/quantizer/gollek-quantizer-gptq")
includeOptionalProject("quantizer:gollek-quantizer-quip", "runtime/quantizer/gollek-quantizer-quip")
includeOptionalProject("quantizer:gollek-quantizer-turboquant", "runtime/quantizer/gollek-quantizer-turboquant")

// Optional runners and runtimes
includeOptionalProject("runner:gollek-diffusion", "runtime/runner/gollek-diffusion")
includeOptionalProject("runtime:gollek-runtime-core", "runtime/core/gollek-runtime-core")
// includeOptionalProject("runtime:distributed:gollek-runtime-distributed", "runtime/core/gollek-runtime-distributed")

// Optional SDK
includeOptionalProject("sdk:gollek-sdk", "framework/sdk/gollek-sdk")
includeOptionalProject("sdk:gollek-sdk-api", "framework/sdk/gollek-sdk-api")
includeOptionalProject("sdk:gollek-sdk-core", "framework/sdk/gollek-sdk-core")
includeOptionalProject("sdk:gollek-sdk-local", "framework/sdk/gollek-sdk-local")
// includeOptionalProject("sdk:gollek-sdk-remote", "framework/sdk/gollek-sdk-remote")
includeOptionalProject("sdk:gollek-sdk-protobuf", "framework/sdk/gollek-sdk-protobuf")
includeOptionalProject("ui:gollek-sdk-protobuf", "ui/gollek-sdk-protobuf")
if (file("framework/sdk/gollek-sdk-session").isDirectory) {
    include("sdk:gollek-sdk-session")
}

// Optional SPIs
includeOptionalProject("spi:gollek-spi", "framework/spi/gollek-spi")
includeOptionalProject("spi:gollek-spi-audio", "framework/spi/gollek-spi-audio")
includeOptionalProject("spi:gollek-spi-image", "framework/spi/gollek-spi-image")
includeOptionalProject("spi:gollek-spi-inference", "framework/spi/gollek-spi-inference")
includeOptionalProject("spi:gollek-spi-model", "framework/spi/gollek-spi-model")
includeOptionalProject("spi:gollek-spi-multimodal", "framework/spi/gollek-spi-multimodal")
includeOptionalProject("spi:gollek-spi-plugin", "framework/spi/gollek-spi-plugin")
includeOptionalProject("spi:gollek-spi-runtime", "framework/spi/gollek-spi-runtime")

// Optional UI
includeOptionalProject("ui:gollek-api", "ui/gollek-api")
includeOptionalProject("ui:gollek-cli", "ui/gollek-cli")

// Optional backends
includeOptionalProject("backend:blackwell:gollek-kernel-blackwell", "runtime/backend/blackwell/gollek-kernel-blackwell")
includeOptionalProject("backend:blackwell:gollek-plugin-kernel-blackwell", "runtime/backend/blackwell/gollek-plugin-kernel-blackwell")
includeOptionalProject("backend:cpu:gollek-backend-cpu", "runtime/backend/cpu/gollek-backend-cpu")
includeOptionalProject("backend:cuda:gollek-backend-cuda", "runtime/backend/cuda/gollek-backend-cuda")
includeOptionalProject("backend:cuda:gollek-kernel-cuda", "runtime/backend/cuda/gollek-kernel-cuda")
includeOptionalProject("backend:cuda:gollek-plugin-kernel-cuda", "runtime/backend/cuda/gollek-plugin-kernel-cuda")
includeOptionalProject("backend:directml:gollek-plugin-kernel-directml", "runtime/backend/directml/gollek-plugin-kernel-directml")
includeOptionalProject("backend:metal:gollek-backend-metal", "runtime/backend/metal/gollek-backend-metal")
includeOptionalProject("backend:metal:gollek-mlx-binding", "runtime/backend/metal/gollek-mlx-binding")
includeOptionalProject("backend:rocm:gollek-kernel-rocm", "runtime/backend/rocm/gollek-kernel-rocm")
includeOptionalProject("backend:rocm:gollek-plugin-kernel-rocm", "runtime/backend/rocm/gollek-plugin-kernel-rocm")

// Optional core plugins
includeOptionalProject("plugin:gollek-plugin-core", "framework/plugin/gollek-plugin-core")
includeOptionalProject("plugin:gollek-plugin-kernel-core", "framework/plugin/gollek-plugin-kernel-core")
includeOptionalProject("plugin:gollek-plugin-optimization-core", "framework/plugin/gollek-plugin-optimization-core")
includeOptionalProject("plugin:gollek-plugin-runner-core", "framework/plugin/gollek-plugin-runner-core")
includeOptionalProject("plugin:gollek-plugin-runner-gguf", "framework/plugin/gollek-plugin-runner-gguf")

// Optional runners
includeOptionalProject("runner:diffuser:gollek-diffuser", "runtime/runner/diffuser/gollek-diffuser")
includeOptionalProject("runner:gguf:gollek-gguf-converter", "runtime/runner/gguf/gollek-gguf-converter")
includeOptionalProject("runner:gguf:gollek-gguf-converter-java", "runtime/runner/gguf/gollek-gguf-converter-java")
includeOptionalProject("runner:gguf:gollek-gguf-core", "runtime/runner/gguf/gollek-gguf-core")
includeOptionalProject("runner:gguf:gollek-runner-gguf", "runtime/runner/gguf/gollek-runner-gguf")
includeOptionalProject("runner:gguf:gollek-plugin-runner-gguf", "runtime/runner/gguf/gollek-plugin-runner-gguf")
includeOptionalProject("runner:gguf:gollek-gguf-feature-text", "runtime/runner/gguf/gollek-gguf-feature-text")
includeOptionalProject("runner:litert:gollek-litert-core", "runtime/runner/litert/gollek-litert-core")
includeOptionalProject("runner:litert:gollek-plugin-runner-litert", "runtime/runner/litert/gollek-plugin-runner-litert")
includeOptionalProject("runner:litert:gollek-runner-litert", "runtime/runner/litert/gollek-runner-litert")

includeOptionalProject("runner:onnx:gollek-plugin-runner-onnx", "runtime/runner/onnx/gollek-plugin-runner-onnx")
includeOptionalProject("runner:onnx:gollek-runner-onnx", "runtime/runner/onnx/gollek-runner-onnx")

includeOptionalProject("runner:safetensor:gollek-safetensor-engine", "runtime/runner/safetensor/gollek-safetensor-engine")
includeOptionalProject("runner:safetensor:gollek-runner-flux", "runtime/runner/safetensor/gollek-runner-flux")
// includeOptionalProject("runner:safetensor:gollek-runner-safetensor", "runtime/runner/safetensor/gollek-runner-safetensor")
// includeOptionalProject("runner:safetensor:gollek-runner-stable-diffusion", "runtime/runner/safetensor/gollek-runner-stable-diffusion")
// includeOptionalProject("runner:safetensor:gollek-safetensor-api", "runtime/runner/safetensor/gollek-safetensor-api")
// includeOptionalProject("runner:safetensor:gollek-safetensor-core", "runtime/runner/safetensor/gollek-safetensor-core")
// includeOptionalProject("runner:safetensor:gollek-safetensor-loader", "runtime/runner/safetensor/gollek-safetensor-loader")
// includeOptionalProject("runner:safetensor:gollek-safetensor-quantization", "runtime/runner/safetensor/gollek-safetensor-quantization")
// includeOptionalProject("runner:safetensor:gollek-safetensor-spi", "runtime/runner/safetensor/gollek-safetensor-spi")
// includeOptionalProject("runner:tensorrt:gollek-runner-tensorrt", "runtime/runner/tensorrt/gollek-runner-tensorrt")
// includeOptionalProject("runner:tensorrt:gollek-plugin-runner-tensorrt", "runtime/runner/tensorrt/gollek-plugin-tensorrt")
// includeOptionalProject("runner:torch:gollek-runner-libtorch", "runtime/runner/torch/gollek-runner-libtorch")





includeBuild("../../Modules/suling")
