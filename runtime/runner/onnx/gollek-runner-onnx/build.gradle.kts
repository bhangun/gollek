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
    modularity.inferModulePath.set(false)
}

repositories {
    mavenCentral()
    mavenLocal()
}

dependencies {
    implementation("tech.kayys.alkhawarizm:alkhawarizm-error-code:0.1.0-SNAPSHOT")
    implementation("tech.kayys.alkhawarizm:alkhawarizm-error-code:0.1.0-SNAPSHOT")
    implementation("tech.kayys.alkhawarizm:alkhawarizm-spi-model:0.1.0-SNAPSHOT")
    implementation("tech.kayys.alkhawarizm:alkhawarizm-tensor:0.1.0-SNAPSHOT")
    implementation(project(":spi:gollek-spi"))
    implementation(project(":spi:gollek-spi-image"))
    implementation(project(":spi:gollek-spi-runner"))
    implementation("tech.kayys.alkhawarizm:alkhawarizm-model-flux:0.1.0-SNAPSHOT")
    implementation("tech.kayys:suling:0.1.0")
    implementation(project(":spi:gollek-spi-multimodal"))
    implementation(project(":core:gollek-core"))
    implementation(project(":spi:gollek-spi-inference"))
    implementation(project(":core:gollek-tokenizer-core"))
    implementation(project(":optimization:gollek-plugin-kv-cache"))
    implementation(group = "com.fasterxml.jackson.core", name = "jackson-databind")
    implementation(group = "com.microsoft.onnxruntime", name = "onnxruntime", version = "1.19.2")
    implementation(group = "io.quarkus", name = "quarkus-arc")
    implementation(group = "io.smallrye.reactive", name = "mutiny")
    compileOnly(group = "org.jboss.logging", name = "jboss-logging")
    testImplementation(group = "org.junit.jupiter", name = "junit-jupiter")
    testImplementation(project(":core:gollek-model-repo-hf"))
    testImplementation(project(":core:gollek-core"))
    testImplementation(project(":core:gollek-core"))
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

tasks.withType<Test>().configureEach {
    useJUnitPlatform()
}
