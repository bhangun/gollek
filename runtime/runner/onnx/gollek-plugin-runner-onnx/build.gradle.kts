plugins {
    `java-library`
    `maven-publish`
}

group = "tech.kayys.gollek"
version = "0.1.0-SNAPSHOT"

java {
    toolchain {
        languageVersion.set(JavaLanguageVersion.of(25))
    }
}

repositories {
    mavenCentral()
    mavenLocal()
}

dependencies {
    implementation(project(":core:plugin:gollek-plugin-runner-core"))
    implementation(project(":runner:onnx:gollek-runner-onnx"))
    implementation(project(":core:gollek-tokenizer-core"))
    implementation(project(":spi:gollek-spi-inference"))
    implementation(project(":spi:gollek-spi-multimodal"))
    compileOnly(group = "io.smallrye.reactive", name = "mutiny")
    implementation(group = "com.microsoft.onnxruntime", name = "onnxruntime", version = "1.20.0")
    compileOnly(group = "org.jboss.logging", name = "jboss-logging")
    testImplementation(group = "org.junit.jupiter", name = "junit-jupiter")
}

publishing {
    publications {
        create<MavenPublication>("mavenJava") {
            from(components["java"])
        }
    }
    repositories {
        mavenLocal()
    }
}

tasks.jar {
    manifest {
        attributes(
            mapOf(
                "Plugin-Author" to "Gollek Team",
                "Plugin-Capabilities" to "onnx-inference, cpu-inference, gpu-inference, cross-platform",
                "Plugin-Deployment" to "standalone,microservice,hybrid",
                "Plugin-Id" to "onnx-runner",
                "Plugin-License" to "MIT",
                "Plugin-Name" to "ONNX Runtime Runner",
                "Plugin-Performance-Memory-Overhead" to "80-150MB",
                "Plugin-Performance-Speedup" to "1-2x (vs baseline)",
                "Plugin-Provider" to "tech.kayys.gollek.plugin.runner.onnx.OnnxRunnerPlugin",
                "Plugin-Type" to "runner",
                "Plugin-Vendor" to "Kayys.tech",
                "Plugin-Version" to "0.1.0-SNAPSHOT",
                "Supported-Architectures" to "bert,roberta,distilbert,whisper,clip,yolo,resnet,vit,llama,mistral",
                "Supported-Formats" to ".onnx,.onnxruntime"
            )
        )
    }
}

tasks.withType<Test>().configureEach {
    useJUnitPlatform()
}
