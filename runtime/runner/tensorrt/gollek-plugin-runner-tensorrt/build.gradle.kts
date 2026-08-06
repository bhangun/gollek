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
    implementation(project(":runner:tensorrt:gollek-runner-tensorrt"))
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
                "Plugin-Id" to "tensorrt-runner",
                "Plugin-Type" to "runner",
                "Plugin-Provider" to "tech.kayys.gollek.plugin.runner.tensorrt.TensorRTRunnerPlugin",
                "Plugin-Version" to "0.1.0-SNAPSHOT",
                "Supported-Formats" to ".engine,.plan"
            )
        )
    }
}

tasks.withType<Test>().configureEach {
    useJUnitPlatform()
}
