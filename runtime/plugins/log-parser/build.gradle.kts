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
    implementation(group = "io.quarkus", name = "quarkus-rest")
    implementation(group = "io.quarkus", name = "quarkus-rest-jackson")
    implementation(group = "io.quarkus", name = "quarkus-vertx")
    implementation(group = "io.quarkus", name = "quarkus-picocli")
    implementation(group = "io.quarkus", name = "quarkus-jackson")
    implementation(group = "io.quarkus", name = "quarkus-arc")
    testImplementation(group = "io.quarkus", name = "quarkus-junit5")
    testImplementation(group = "io.rest-assured", name = "rest-assured")
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

tasks.withType<GenerateModuleMetadata> {
    suppressedValidationErrors.add("dependencies-without-versions")
}
