plugins {
    java
}

dependencies {
    implementation(project(":runner:gguf:gollek-gguf-core"))
    implementation(project(":core:gollek-core"))
    implementation("tech.kayys.alkhawarizm:alkhawarizm-tensor:0.1.0-SNAPSHOT")
    implementation(project(":spi:gollek-spi"))
    implementation(project(":runner:safetensor:gollek-safetensor-loader"))
    implementation("io.smallrye.reactive:mutiny:2.5.5")
    implementation("com.google.code.gson:gson:2.11.0")
    implementation("com.fasterxml.jackson.core:jackson-annotations:2.16.1")
    implementation("com.fasterxml.jackson.core:jackson-databind:2.16.1")
    implementation("jakarta.enterprise:jakarta.enterprise.cdi-api")
    implementation("jakarta.inject:jakarta.inject-api")
    implementation("org.eclipse.microprofile.config:microprofile-config-api:3.1")
    implementation("org.slf4j:slf4j-api:2.0.13")

    testImplementation("org.junit.jupiter:junit-jupiter")
    testImplementation("org.assertj:assertj-core:3.26.3")
    testImplementation("io.quarkus:quarkus-junit5")
}

tasks.withType<Test>().configureEach {
    useJUnitPlatform()
}

tasks.register<JavaExec>("ggufConvert") {
    group = "gguf"
    description = "Convert a local HuggingFace/safetensors model directory to GGUF."
    classpath = sourceSets["main"].runtimeClasspath
    mainClass.set("tech.kayys.gollek.converter.gguf.GgufConverterMain")

    doFirst {
        val input = providers.gradleProperty("ggufInput").orNull
            ?: error("Missing -PggufInput=/path/to/model-dir")
        val output = providers.gradleProperty("ggufOutput").orNull
            ?: error("Missing -PggufOutput=/path/to/output.gguf")
        val type = providers.gradleProperty("ggufType").orElse("F16").get()
        val version = providers.gradleProperty("ggufVersion").orElse("1.0").get()
        val verbose = providers.gradleProperty("ggufVerbose").map(String::toBoolean).orElse(false).get()

        args("convert", input, output, "--type", type, "--version", version)
        if (verbose) {
            args("--verbose")
        }
    }
}
