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
    implementation("io.smallrye.reactive:mutiny:2.5.5")


    implementation(project(":spi:gollek-spi"))
    implementation(project(":spi:gollek-spi-plugin"))
    implementation(project(":core:plugin:gollek-plugin-optimization-core"))

    implementation(project(":core:gollek-model-runner"))
    implementation(project(":core:gollek-error-code"))
    implementation(project(":core:gollek-core"))
    implementation("tech.kayys.alkhawarizm:alkhawarizm-tensor:0.1.0-SNAPSHOT")
    implementation("tech.kayys.alkhawarizm:alkhawarizm-core:0.1.0-SNAPSHOT")

    implementation("tech.kayys.alkhawarizm:alkhawarizm-spi-model:0.1.0-SNAPSHOT")
    implementation(project(":optimization:gollek-plugin-kv-cache"))
    
    implementation(group = "io.quarkus", name = "quarkus-rest")
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
