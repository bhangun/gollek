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
    implementation("tech.kayys.alkhawarizm:alkhawarizm-gguf-core:0.1.0-SNAPSHOT")
    implementation("tech.kayys.alkhawarizm:alkhawarizm-nn:0.1.0")
    implementation("tech.kayys.alkhawarizm:alkhawarizm-tensor:0.1.0")
    implementation(project(":spi:gollek-spi-inference"))
    implementation(project(":runner:safetensor:gollek-safetensor-engine"))
    testImplementation(group = "org.junit.jupiter", name = "junit-jupiter")
    testRuntimeOnly(group = "org.junit.platform", name = "junit-platform-launcher")
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
                "Plugin-Id" to "gguf-runner",
                "Plugin-Type" to "runner",
                "Plugin-Provider" to "tech.kayys.gollek.plugin.runner.gguf.GgufRunnerPlugin",
                "Plugin-Version" to "0.1.0-SNAPSHOT",
                "Supported-Formats" to ".gguf"
            )
        )
    }
}

tasks.withType<Test>().configureEach {
    useJUnitPlatform()
}
