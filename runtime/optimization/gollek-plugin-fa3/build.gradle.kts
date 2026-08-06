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
    implementation(project(":core:plugin:gollek-plugin-optimization-core"))
    implementation(group = "net.java.dev.jna", name = "jna", version = "5.14.0")
    implementation(project(":spi:gollek-spi-inference"))
    implementation(group = "io.quarkus", name = "quarkus-arc", version = "3.32.2")
    implementation(group = "io.smallrye.reactive", name = "mutiny", version = "2.9.4")
    compileOnly(group = "org.jboss.logging", name = "jboss-logging", version = "3.6.1.Final")
    testImplementation(group = "org.junit.jupiter", name = "junit-jupiter", version = "5.11.4")
    testImplementation(group = "org.assertj", name = "assertj-core", version = "3.27.3")
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
