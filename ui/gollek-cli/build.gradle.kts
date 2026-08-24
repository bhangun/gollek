plugins {
    java
    id("io.quarkus")
}

// Prevent old tech.kayys.alkhawarizm:gollek-spi* published snapshots from landing on the
// classpath alongside the live project modules.  Both carry identical Quarkus CDI extensions
// that would register the same synthetic beans (e.g. KaggleConfig) twice, causing:
//   IllegalStateException: A synthetic bean … is already registered
configurations.all {
    exclude(group = "tech.kayys.alkhawarizm", module = "gollek-spi")
    exclude(group = "tech.kayys.alkhawarizm", module = "gollek-spi-inference")
    exclude(group = "tech.kayys.alkhawarizm", module = "gollek-spi-multimodal")
    exclude(group = "tech.kayys.alkhawarizm", module = "gollek-spi-plugin")
    exclude(group = "tech.kayys.alkhawarizm", module = "gollek-spi-runtime")
    
    // Also exclude obsolete tech.kayys.gollek artifacts that have been merged/refactored
    // (e.g. gollek-runtime-config was split into gollek-model-repo-hf and others)
    exclude(group = "tech.kayys.gollek", module = "gollek-runtime-config")
    exclude(group = "tech.kayys.gollek", module = "gollek-ir")
    
    // Exclude obsolete safetensor artifacts in the old namespace to avoid CDI issues
    // where old engines (like WhisperEngine) try to inject the old excluded SPI.
    exclude(group = "tech.kayys.gollek", module = "gollek-safetensor-audio")
    exclude(group = "tech.kayys.gollek", module = "gollek-safetensor-api")
    exclude(group = "tech.kayys.gollek", module = "gollek-safetensor-core")
    exclude(group = "tech.kayys.gollek", module = "gollek-safetensor-loader")
    exclude(group = "tech.kayys.gollek", module = "gollek-safetensor-spi")
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
    implementation(project(":sdk:gollek-sdk-core"))
    implementation(project(":spi:gollek-spi"))
    implementation(project(":spi:gollek-spi-inference"))
    implementation("tech.kayys.alkhawarizm:alkhawarizm-spi-model:0.1.0-SNAPSHOT")
    implementation(project(":spi:gollek-spi-multimodal"))
    implementation(project(":spi:gollek-spi-runtime"))
    implementation(project(":core:gollek-core"))
    implementation(project(":core:gollek-tokenizer-core"))
    implementation("io.opentelemetry:opentelemetry-api")
    implementation(project(":plugin:gollek-plugin-core"))
    implementation(project(":plugin:gollek-plugin-kernel-core"))
    implementation(project(":plugin:gollek-plugin-runner-core"))
    implementation(project(":plugin:gollek-plugin-runner-gguf"))
    implementation(project(":observability:gollek-observability"))
    implementation(project(":core:gollek-model-repo-hf"))
    implementation(project(":core:gollek-model-repo-kaggle"))
    implementation(project(":core:gollek-model-repo-local"))

    implementation(project(":plugins:log-parser"))
    implementation(project(":runner:litert:gollek-runner-litert"))
    implementation(project(":runner:onnx:gollek-runner-onnx"))
    implementation("tech.kayys.alkhawarizm:alkhawarizm-gguf-core:0.1.0-SNAPSHOT")
    implementation(project(":suling"))
    implementation("com.google.ai.edge.litertlm:litertlm-jvm:0.11.0")
    implementation(project(":runner:safetensor:gollek-safetensor-engine"))
    implementation("tech.kayys.alkhawarizm:alkhawarizm-safetensor-loader:0.1.0-SNAPSHOT")
    implementation("tech.kayys.alkhawarizm:alkhawarizm-safetensor-spi:0.1.0-SNAPSHOT")
    implementation("tech.kayys.alkhawarizm:alkhawarizm-safetensor-quantization:0.1.0-SNAPSHOT")
    implementation("tech.kayys.alkhawarizm:alkhawarizm-backend-metal:0.1.0-SNAPSHOT")
    implementation("tech.kayys.alkhawarizm:alkhawarizm-model-gemma:0.1.0-SNAPSHOT")
    implementation("tech.kayys.alkhawarizm:alkhawarizm-model-qwen:0.1.0-SNAPSHOT")

    testImplementation(platform("org.junit:junit-bom:5.10.0"))
    testImplementation("org.junit.jupiter:junit-jupiter")
    testImplementation("io.quarkus:quarkus-junit5")
    testImplementation("io.quarkus:quarkus-junit5-mockito")
}

sourceSets {
    main {
        java {
            setSrcDirs(listOf("src/main/java"))
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
