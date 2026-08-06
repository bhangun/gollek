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
    implementation(group = "net.java.dev.jna", name = "jna", version = "5.14.0")
    implementation(group = "org.jboss.logging", name = "jboss-logging", version = "3.6.0.Final")
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
                "Plugin-Class" to "tech.kayys.gollek.plugin.optimization.OptimizationPluginManager",
                "Plugin-Id" to "optimization-core",
                "Plugin-Type" to "optimization-manager",
                "Plugin-Version" to "0.1.0-SNAPSHOT"
            )
        )
    }
}
