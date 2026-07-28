plugins {
    `java-library`
    `maven-publish`
}

group = "tech.kayys.gollek"
version = "0.1.0-SNAPSHOT"

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


dependencies {
    implementation(project(":spi:gollek-spi-inference"))
    implementation(project(":core:plugin:gollek-plugin-runner-core"))
    implementation(project(":plugins:gollek-plugin-model-router"))
    implementation(project(":core:plugin:gollek-plugin-optimization-core"))
    implementation("io.smallrye.reactive:mutiny")
    
    // GraalVM annotations for NativeImageFeature
    compileOnly("org.graalvm.sdk:nativeimage:24.1.2")
    implementation("io.quarkus:quarkus-core")

    // Cache plugin required for SemanticCachePlugin
    implementation(project(":plugins:gollek-plugin-semantic-cache"))

    testImplementation("org.junit.jupiter:junit-jupiter")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
}

tasks.withType<Test>().configureEach {
    useJUnitPlatform()
}
