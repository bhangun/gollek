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
    implementation("io.smallrye.reactive:mutiny:2.5.5")
    implementation("org.jboss.logging:jboss-logging:3.6.1.Final")
    implementation(project(":optimization:gollek-plugin-kv-cache"))
    implementation(project(":core:plugin:gollek-plugin-optimization-core"))
    implementation(project(":spi:gollek-spi"))
    compileOnly(group = "jakarta.enterprise", name = "jakarta.enterprise.cdi-api")
    compileOnly(group = "jakarta.inject", name = "jakarta.inject-api")
    compileOnly(group = "org.jboss.logging", name = "jboss-logging")
    testImplementation(group = "org.junit.jupiter", name = "junit-jupiter")
    testImplementation(group = "org.assertj", name = "assertj-core")
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
