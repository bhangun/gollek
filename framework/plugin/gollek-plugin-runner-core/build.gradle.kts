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
    api(project(":spi:gollek-spi"))
    api(project(":spi:gollek-spi-inference"))
    api("tech.kayys.alkhawarizm:alkhawarizm-spi-model:0.1.0-SNAPSHOT")
    implementation("org.jboss.logging:jboss-logging:3.6.0.Final")
    implementation("io.smallrye.reactive:mutiny:2.5.5")
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
                "Plugin-Class" to "tech.kayys.gollek.plugin.runner.RunnerPluginManager",
                "Plugin-Id" to "runner-core",
                "Plugin-Type" to "runner-manager",
                "Plugin-Version" to "0.1.0-SNAPSHOT"
            )
        )
    }
}
