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
    implementation(project(":spi:gollek-spi-inference"))
    implementation(project(":runner:safetensor:gollek-safetensor-spi"))
    implementation(project(":runner:safetensor:gollek-safetensor-engine"))
    implementation(project(":runner:safetensor:gollek-safetensor-core"))
    implementation(project(":runner:safetensor:gollek-safetensor-api"))
    implementation(project(":runner:safetensor:gollek-safetensor-quantization"))
    implementation(project(":spi:gollek-spi-model"))
    implementation(project(":spi:gollek-spi-multimodal"))
    implementation(project(":core:gollek-runtime-config"))
    implementation("tech.kayys.tafkir:tafkir-quantizer-gptq:0.1.0-SNAPSHOT")
    implementation("com.fasterxml.jackson.core:jackson-annotations:2.16.1")
    implementation("com.fasterxml.jackson.core:jackson-databind:2.16.1")
    implementation("io.smallrye.reactive:mutiny:2.5.5")
    implementation("jakarta.ws.rs:jakarta.ws.rs-api:3.1.0")
    implementation("jakarta.enterprise:jakarta.enterprise.cdi-api:4.0.1")
    implementation("org.jboss.logging:jboss-logging:3.5.3.Final")
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
