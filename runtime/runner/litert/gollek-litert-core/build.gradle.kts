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
    implementation("tech.kayys.alkhawarizm:alkhawarizm-nn:0.1.0")
    implementation("tech.kayys.alkhawarizm:alkhawarizm-tensor:0.1.0")
    implementation(project(":sdk:gollek-sdk-api"))
    implementation(project(":runner:litert:gollek-runner-litert"))
    implementation(project(":spi:gollek-spi-inference"))
    implementation(group = "com.fasterxml.jackson.core", name = "jackson-databind")
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

tasks.withType<Test>().configureEach {
    useJUnitPlatform()
}
