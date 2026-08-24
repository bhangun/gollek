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

    implementation(project(":spi:gollek-spi"))
    implementation(project(":spi:gollek-spi-plugin"))
    implementation(project(":plugin:gollek-plugin-optimization-core"))

    implementation(project(":spi:gollek-spi-inference"))
    implementation(project(":optimization:gollek-plugin-kv-cache"))
    
    implementation(group = "io.quarkus", name = "quarkus-arc")
    implementation(group = "io.quarkus", name = "quarkus-config-yaml")
    implementation(group = "io.smallrye.config", name = "smallrye-config")
    implementation(group = "io.quarkus", name = "quarkus-rest-jackson")
    implementation(group = "io.quarkus", name = "quarkus-smallrye-openapi")
    implementation(group = "com.github.ben-manes.caffeine", name = "caffeine")
    implementation(group = "io.quarkus", name = "quarkus-redis-client")
    implementation(group = "com.fasterxml.jackson.core", name = "jackson-databind")
    implementation(group = "com.fasterxml.jackson.datatype", name = "jackson-datatype-jsr310")
    implementation(group = "io.quarkus", name = "quarkus-micrometer-registry-prometheus")
    implementation(group = "io.quarkus", name = "quarkus-smallrye-health")
    implementation(group = "org.jboss.logging", name = "jboss-logging")
    testImplementation(group = "io.quarkus", name = "quarkus-junit")
    testImplementation(group = "org.junit.jupiter", name = "junit-jupiter")
    implementation(group = "io.micrometer", name = "micrometer-core")
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
