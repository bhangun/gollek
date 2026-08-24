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
    implementation(project(":spi:gollek-spi-inference"))
    implementation(project(":core:gollek-core"))
    implementation("tech.kayys.alkhawarizm:alkhawarizm-error-code:0.1.0-SNAPSHOT")
    implementation(project(":core:gollek-core"))
    implementation("tech.kayys.alkhawarizm:alkhawarizm-tensor:0.1.0-SNAPSHOT")
    implementation("tech.kayys.alkhawarizm:alkhawarizm-core:0.1.0-SNAPSHOT")

    implementation("tech.kayys.alkhawarizm:alkhawarizm-spi-model:0.1.0-SNAPSHOT")
    implementation(project(":optimization:gollek-plugin-kv-cache"))
    implementation(project(":optimization:gollek-plugin-paged-attention"))
    implementation(group = "io.quarkus", name = "quarkus-arc")
    compileOnly(group = "org.jboss.logging", name = "jboss-logging")
    implementation(group = "io.smallrye.reactive", name = "mutiny")
    implementation(project(":core:gollek-core"))
    implementation("tech.kayys.alkhawarizm:alkhawarizm-error-code:0.1.0-SNAPSHOT")
    implementation(project(":core:gollek-core"))
    implementation("tech.kayys.alkhawarizm:alkhawarizm-tensor:0.1.0-SNAPSHOT")
    implementation("tech.kayys.alkhawarizm:alkhawarizm-core:0.1.0-SNAPSHOT")

    implementation("tech.kayys.alkhawarizm:alkhawarizm-spi-model:0.1.0-SNAPSHOT")
    implementation("tech.kayys.alkhawarizm:alkhawarizm-error-code:0.1.0-SNAPSHOT")
    implementation("tech.kayys.alkhawarizm:alkhawarizm-tensor:0.1.0-SNAPSHOT")
    implementation(project(":core:gollek-core"))
    implementation("tech.kayys.alkhawarizm:alkhawarizm-tensor:0.1.0-SNAPSHOT")
    implementation("tech.kayys.alkhawarizm:alkhawarizm-core:0.1.0-SNAPSHOT")

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
