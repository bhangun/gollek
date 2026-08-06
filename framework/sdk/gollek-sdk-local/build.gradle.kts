plugins {
    java
    `maven-publish`
}

group = "tech.kayys.gollek"
version = "0.1.0-SNAPSHOT"

publishing {
    publications {
        create<MavenPublication>("mavenJava") {
            artifactId = "gollek-sdk-local"
            from(components["java"])
        }
    }
    repositories {
        mavenLocal()
    }
}

dependencies {
    implementation(project(":sdk:gollek-sdk"))
    implementation(project(":sdk:gollek-sdk-core"))
    implementation(project(":sdk:gollek-sdk-api"))
    implementation("io.quarkus:quarkus-arc:3.32.2")
    implementation(project(":core:gollek-core"))
    implementation(project(":core:gollek-model-repository"))
    implementation(project(":core:gollek-observability"))
    implementation("tech.kayys.alkhawarizm:alkhawarizm-tensor:0.1.0-SNAPSHOT")
    implementation(project(":plugins:gollek-plugin-mcp"))
    implementation(project(":plugins:gollek-plugin-model-router"))
    implementation("tech.kayys.alkhawarizm:alkhawarizm-gguf-converter:0.1.0-SNAPSHOT")
    implementation(project(":core:gollek-model-repo-hf"))
    implementation(project(":core:gollek-model-repo-local"))
    implementation(project(":core:gollek-tokenizer-core"))
    implementation(project(":spi:gollek-spi"))
    implementation(project(":spi:gollek-spi-inference"))
    implementation("tech.kayys.alkhawarizm:alkhawarizm-spi-model:0.1.0-SNAPSHOT")
    implementation(project(":spi:gollek-spi-multimodal"))
    implementation(project(":spi:gollek-spi-plugin"))
    implementation("tech.kayys.alkhawarizm:alkhawarizm-gguf-core:0.1.0-SNAPSHOT")
    implementation("tech.kayys.alkhawarizm:alkhawarizm-safetensor-core:0.1.0-SNAPSHOT")
    implementation(project(":runner:safetensor:gollek-safetensor-engine"))
    implementation("tech.kayys.alkhawarizm:alkhawarizm-safetensor-spi:0.1.0-SNAPSHOT")
    implementation(project(":core:gollek-runtime-config"))
    implementation(project(":optimization:gollek-plugin-kv-cache"))



    implementation("io.smallrye.reactive:mutiny:2.5.5")
    implementation("io.opentelemetry:opentelemetry-api:1.34.1")
    implementation("jakarta.enterprise:jakarta.enterprise.cdi-api:4.0.1")
    implementation("jakarta.inject:jakarta.inject-api:2.0.1")
    implementation("jakarta.validation:jakarta.validation-api:3.0.2")
    
    implementation("com.fasterxml.jackson.core:jackson-databind:2.16.1")
    implementation("com.fasterxml.jackson.datatype:jackson-datatype-jsr310:2.16.1")
    implementation("org.apache.commons:commons-lang3:3.14.0")
    implementation("org.slf4j:slf4j-api:2.0.13")
    
    compileOnly("org.graalvm.sdk:nativeimage:24.1.2")
}
