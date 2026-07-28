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

val jacksonVersion = "2.16.1"
val mutinyVersion = "2.5.5"
val jakartaValidationVersion = "3.0.2"

dependencies {
    api(project(":sdk:gollek-sdk-agent"))
    api(project(":sdk:gollek-sdk-core"))
    api(project(":spi:gollek-spi"))
    api(project(":spi:gollek-spi-inference"))
    api(project(":spi:gollek-spi-model"))
    api(project(":spi:gollek-spi-multimodal"))
    implementation(group = "com.fasterxml.jackson.core", name = "jackson-databind", version = jacksonVersion)
    implementation(group = "io.smallrye.reactive", name = "mutiny", version = mutinyVersion)
    implementation(group = "io.reactivex.rxjava3", name = "rxjava", version = "3.1.8")
    implementation(group = "jakarta.validation", name = "jakarta.validation-api", version = jakartaValidationVersion)
    testImplementation(group = "org.junit.jupiter", name = "junit-jupiter")
    testImplementation(group = "org.mockito", name = "mockito-core")
    testImplementation(group = "org.mockito", name = "mockito-junit-jupiter")
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
                "Automatic-Module-Name" to "tech.kayys.gollek.sdk.java"
            )
        )
    }
}

tasks.withType<Test>().configureEach {
    useJUnitPlatform()
}
