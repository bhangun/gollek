import os

build_gradle_path = './runner/torch/gollek-runner-libtorch/build.gradle.kts'
content = '''plugins {
    `java-library`
    `maven-publish`
}

group = "tech.kayys.gollek"
version = "0.1.0-SNAPSHOT"

val quarkusVersion = rootProject.extra["quarkusVersion"] as String

java {
    toolchain {
        languageVersion.set(JavaLanguageVersion.of(25))
    }
}

sourceSets {
    named("main") {
        java {
            setSrcDirs(listOf("src/main/java"))

            // Keep Gradle on the maintained LibTorch binding/configuration slice first.
            // The broader tensor/NN/provider runtime still depends on removed
            // runtime.* / engine.* package roots and migrates separately.
            include("tech/kayys/gollek/inference/libtorch/binding/**")
            include("tech/kayys/gollek/inference/libtorch/config/**")
            include("tech/kayys/gollek/inference/libtorch/plugin/**")
            include("tech/kayys/gollek/inference/libtorch/util/SafetensorsHeaderParser.java")
            // Temporarily disable LibTorchProvider and related classes from compilation until they migrate to core.tensor APIs
            // include("tech/kayys/gollek/inference/libtorch/LibTorchAdapterApplier.java")
            // include("tech/kayys/gollek/inference/libtorch/LibTorchAdvancedModeResolver.java")
            // include("tech/kayys/gollek/inference/libtorch/LibTorchBeanProducer.java")
            // include("tech/kayys/gollek/inference/libtorch/LibTorchChatTemplateService.java")
            // include("tech/kayys/gollek/inference/libtorch/LibTorchDeviceSupport.java")
            // include("tech/kayys/gollek/inference/libtorch/LibTorchExecutionHints.java")
            // include("tech/kayys/gollek/inference/libtorch/LibTorchFp8CalibrationLoader.java")
            // include("tech/kayys/gollek/inference/libtorch/LibTorchFp8CalibrationValidator.java")
            // include("tech/kayys/gollek/inference/libtorch/LibTorchFp8RowwisePlanner.java")
            // include("tech/kayys/gollek/inference/libtorch/LibTorchFp8RowwiseTransformer.java")
            // include("tech/kayys/gollek/inference/libtorch/LibTorchGenerationParams.java")
            // include("tech/kayys/gollek/inference/libtorch/LibTorchMetrics.java")
            // include("tech/kayys/gollek/inference/libtorch/LibTorchProvider.java")
            // include("tech/kayys/gollek/inference/libtorch/LibTorchTokenizerManager.java")
            include("tech/kayys/gollek/inference/libtorch/core/Device.java")
        }
    }
}

dependencies {
    implementation(platform("io.quarkus.platform:quarkus-bom:$quarkusVersion"))

    implementation(project(":spi:gollek-spi"))
    implementation("tech.kayys.alkhawarizm:alkhawarizm-tensor:0.1.0-SNAPSHOT")
    implementation(project(":core:gollek-tokenizer-core"))

    implementation("io.quarkus:quarkus-arc")
    implementation("jakarta.json:jakarta.json-api")
    implementation("io.smallrye.config:smallrye-config:3.10.1")
    implementation("io.micrometer:micrometer-core:1.14.4")
    implementation("org.jboss.logging:jboss-logging:3.6.1.Final")

    testImplementation(platform("org.junit:junit-bom:5.10.0"))
    testImplementation("org.junit.jupiter:junit-jupiter")
    testImplementation("org.mockito:mockito-junit-jupiter:5.14.2")
    testImplementation("org.assertj:assertj-core:3.25.3")
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
'''
with open(build_gradle_path, 'w') as f:
    f.write(content)
