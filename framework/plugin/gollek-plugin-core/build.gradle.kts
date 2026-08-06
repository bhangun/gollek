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
    implementation(group = "io.quarkus", name = "quarkus-arc")
    implementation(project(":spi:gollek-spi"))
    implementation(project(":spi:gollek-spi-plugin"))
    implementation(project(":spi:gollek-spi-inference"))
    implementation(group = "com.fasterxml.jackson.core", name = "jackson-databind")
    implementation(group = "org.slf4j", name = "slf4j-api")
    implementation(group = "org.apache.maven", name = "maven-model", version = "3.9.9")
    implementation(group = "org.apache.maven", name = "maven-model-builder", version = "3.9.9")
    implementation(group = "org.apache.maven.resolver", name = "maven-resolver-api", version = "1.9.25")
    implementation(group = "org.apache.maven.resolver", name = "maven-resolver-spi", version = "1.9.25")
    implementation(group = "org.apache.maven.resolver", name = "maven-resolver-util", version = "1.9.25")
    implementation(group = "org.apache.maven.resolver", name = "maven-resolver-impl", version = "1.9.25")
    implementation(group = "org.apache.maven.resolver", name = "maven-resolver-supplier", version = "1.9.25")
    implementation(group = "org.apache.maven.resolver", name = "maven-resolver-connector-basic", version = "1.9.25")
    implementation(group = "org.apache.maven.resolver", name = "maven-resolver-transport-file", version = "1.9.25")
    implementation(group = "org.apache.maven.resolver", name = "maven-resolver-transport-http", version = "1.9.25")
    implementation(group = "org.apache.maven", name = "maven-resolver-provider", version = "3.9.9")
    testImplementation(group = "io.quarkus", name = "quarkus-junit")
    testImplementation(group = "org.junit.jupiter", name = "junit-jupiter")
}

tasks.test {
    useJUnitPlatform()
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
