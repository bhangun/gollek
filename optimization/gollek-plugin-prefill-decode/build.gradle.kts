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
    implementation(project(":spi:gollek-spi-inference"))
    implementation(project(":core:gollek-model-runner"))
    implementation(project(":core:gollek-error-code"))
    implementation(project(":core:gollek-core"))
    implementation("tech.kayys.aljabr:aljabr-tensor:0.1.0-SNAPSHOT")
    implementation("tech.kayys.aljabr:aljabr-core:0.1.0-SNAPSHOT")

    implementation(project(":spi:gollek-spi-model"))
    implementation(project(":optimization:gollek-plugin-kv-cache"))
    implementation(project(":optimization:gollek-plugin-paged-attention"))
    implementation(project(":core:gollek-tokenizer-core"))
    implementation(group = "io.quarkus", name = "quarkus-rest")
    implementation(group = "io.smallrye.reactive", name = "mutiny")
    implementation(group = "io.quarkus", name = "quarkus-arc")
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

tasks.withType<Test>().configureEach {
    useJUnitPlatform()
}
