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
    implementation("tech.kayys.alkhawarizm:alkhawarizm-spi-model:0.1.0-SNAPSHOT")
    implementation(project(":spi:gollek-spi-inference"))
    implementation(project(":spi:gollek-spi"))
    implementation("tech.kayys.alkhawarizm:alkhawarizm-error-code:0.1.0-SNAPSHOT")
    implementation("tech.kayys.alkhawarizm:alkhawarizm-tensor:0.1.0-SNAPSHOT")
    implementation(project(":core:gollek-core"))
    implementation(group = "org.slf4j", name = "slf4j-api")
    compileOnly(group = "org.jboss.logging", name = "jboss-logging")
    implementation(group = "io.quarkus", name = "quarkus-arc")
    compileOnly(group = "org.jetbrains", name = "annotations", version = "24.1.0")
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
