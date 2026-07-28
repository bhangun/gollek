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
    implementation(project(":core:gollek-error-code"))
    implementation("tech.kayys.aljabr:aljabr-error-code:0.1.0-SNAPSHOT")
    implementation("tech.kayys.aljabr:aljabr-spi-model:0.1.0-SNAPSHOT")
    implementation("tech.kayys.aljabr:aljabr-tensor:0.1.0-SNAPSHOT")
    implementation(project(":spi:gollek-spi"))
    implementation(project(":spi:gollek-spi-model"))
    
    implementation(project(":spi:gollek-spi-multimodal"))
    implementation(project(":core:gollek-model-runner"))
    implementation(project(":spi:gollek-spi-inference"))
    implementation(project(":core:gollek-tokenizer-core"))
    implementation(project(":optimization:gollek-plugin-kv-cache"))
    if (findProject(":suling") != null) {
        implementation(project(":suling"))
    }
    implementation(group = "com.fasterxml.jackson.core", name = "jackson-databind")
    implementation(group = "com.microsoft.onnxruntime", name = "onnxruntime", version = "1.19.2")
    implementation(group = "io.quarkus", name = "quarkus-arc")
    implementation(group = "io.smallrye.reactive", name = "mutiny")
    compileOnly(group = "org.jboss.logging", name = "jboss-logging")
    testImplementation(group = "org.junit.jupiter", name = "junit-jupiter")
    testRuntimeOnly(group = "org.junit.platform", name = "junit-platform-launcher")
    testImplementation("tech.kayys.aljabr:aljabr-model-repo-hf:0.1.0-SNAPSHOT")
    testImplementation(project(":core:gollek-model-repository"))
    testImplementation(project(":core:gollek-runtime-config"))
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
