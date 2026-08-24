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
    implementation(group = "org.slf4j", name = "slf4j-api")
    implementation("tech.kayys.alkhawarizm:alkhawarizm-spi-model:0.1.0-SNAPSHOT")
    implementation(project(":spi:gollek-spi"))
    implementation(project(":spi:gollek-spi-inference"))
    implementation(project(":core:gollek-core"))
    implementation("jakarta.validation:jakarta.validation-api:3.0.2")
    implementation(group = "io.quarkus", name = "quarkus-micrometer")
    implementation(group = "io.quarkus", name = "quarkus-micrometer-registry-prometheus")
    implementation(group = "io.quarkus", name = "quarkus-opentelemetry")
    implementation(group = "io.quarkus", name = "quarkus-logging-json")
    implementation(group = "io.quarkus", name = "quarkus-smallrye-health")
    implementation(group = "io.quarkus", name = "quarkus-smallrye-fault-tolerance")
    implementation(group = "io.quarkus", name = "quarkus-cache")
    implementation(group = "io.quarkus", name = "quarkus-config-yaml")
    implementation("io.quarkus:quarkus-rest")
    implementation("io.quarkus:quarkus-rest-jackson")
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
