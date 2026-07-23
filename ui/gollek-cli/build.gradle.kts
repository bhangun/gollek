plugins {
    java
    id("io.quarkus")
}

val quarkusVersion = rootProject.extra["quarkusVersion"] as String

dependencies {
    implementation(platform("io.quarkus.platform:quarkus-bom:$quarkusVersion"))

    implementation("io.quarkus:quarkus-arc")
    implementation("io.quarkus:quarkus-cache")
    implementation("io.quarkus:quarkus-jackson")
    implementation("io.quarkus:quarkus-logging-json")
    implementation("io.quarkus:quarkus-picocli")
    implementation("io.quarkus:quarkus-mutiny")
    implementation("io.quarkus:quarkus-smallrye-health")

    implementation("com.fasterxml.jackson.datatype:jackson-datatype-jsr310")
    implementation("org.apache.commons:commons-lang3:3.14.0")
    implementation("commons-io:commons-io:2.18.0")
    implementation("org.jline:jline-console:3.26.3")
    implementation("org.jline:jline-reader:3.26.3")
    implementation("org.jline:jline-terminal:3.26.3")
    implementation("org.jline:jline-terminal-jna:3.26.3")
    implementation("com.github.albfernandez:juniversalchardet:2.4.0")

    implementation(project(":sdk:gollek-sdk"))
    implementation(project(":sdk:gollek-sdk-api"))
    implementation(project(":sdk:gollek-sdk-agent"))
    implementation(project(":sdk:gollek-sdk-core"))
    implementation(project(":spi:gollek-spi"))
    implementation(project(":spi:gollek-spi-inference"))
    implementation(project(":spi:gollek-spi-model"))
    implementation(project(":spi:gollek-spi-multimodal"))
    implementation(project(":spi:gollek-spi-provider"))
    implementation(project(":spi:gollek-spi-runtime"))
    implementation(project(":core:gollek-model-repository"))
    implementation(project(":core:gollek-runtime-config"))
    implementation(project(":core:gollek-provider-routing"))
    implementation(project(":core:gollek-tokenizer-core"))
    implementation(project(":core:gollek-observability"))
    implementation("io.opentelemetry:opentelemetry-api")
    implementation(project(":core:plugin:gollek-plugin-core"))
    implementation(project(":core:plugin:gollek-plugin-kernel-core"))
    implementation(project(":core:plugin:gollek-plugin-runner-core"))
    implementation(project(":core:plugin:gollek-plugin-runner-gguf"))
    implementation(project(":core:gollek-model-repo-hf"))
    implementation(project(":core:gollek-model-repo-kaggle"))
    implementation(project(":core:gollek-model-repo-local"))
    implementation(project(":plugins:gollek-plugin-mcp"))
    implementation(project(":plugins:log-parser"))
    implementation(project(":runner:litert:gollek-runner-litert"))
    implementation(project(":runner:onnx:gollek-runner-onnx"))
    implementation(project(":runner:gguf:gollek-gguf-core"))
    implementation(project(":suling"))
    implementation("com.google.ai.edge.litertlm:litertlm-jvm:0.11.0")
    implementation(project(":runner:safetensor:gollek-safetensor-engine"))
    implementation(project(":runner:safetensor:gollek-safetensor-loader"))
    implementation(project(":runner:safetensor:gollek-safetensor-spi"))
    implementation(project(":runner:safetensor:gollek-safetensor-quantization"))
    implementation("tech.kayys.aljabr:aljabr-backend-metal:0.1.0-SNAPSHOT")
    implementation(project(":models:gollek-model-gemma4"))
    implementation(project(":models:gollek-model-qwen3-5-moe"))
    implementation(project(":models:gollek-model-qwen3-moe"))

    testImplementation(platform("org.junit:junit-bom:5.10.0"))
    testImplementation("org.junit.jupiter:junit-jupiter")
    testImplementation("io.quarkus:quarkus-junit5")
    testImplementation("io.quarkus:quarkus-junit5-mockito")
}

sourceSets {
    main {
        java {
            setSrcDirs(listOf("src/main/java"))
            exclude("tech/kayys/gollek/cli/GollekCLI.java")
        }
        resources {
            setSrcDirs(listOf("src/main/resources"))
        }
    }
}

tasks.processResources {
    filesMatching("META-INF/gollek-version.properties") {
        filter { line: String ->
            line.replace("\${project.version}", project.version.toString())
        }
    }
}

tasks.jar {
    archiveBaseName.set("gollek")
}

tasks.test {
    useJUnitPlatform()
}
