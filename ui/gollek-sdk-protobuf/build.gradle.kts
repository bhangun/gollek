plugins {
    java
    id("io.quarkus")
}

val quarkusVersion = rootProject.extra["quarkusVersion"] as String

group = "tech.kayys.gollek"
version = "0.1.0-SNAPSHOT"

java {
    toolchain {
        languageVersion.set(JavaLanguageVersion.of(25))
    }
}

dependencies {
    implementation(platform("io.quarkus.platform:quarkus-bom:$quarkusVersion"))
    implementation("io.quarkus:quarkus-grpc")
    implementation(project(":spi:gollek-spi"))
    implementation(project(":spi:gollek-spi-inference"))
    implementation(project(":core:gollek-core"))
    implementation(project(":core:gollek-tokenizer-core"))
    implementation(project(":repository:gollek-model-database"))
    implementation(project(":sdk:gollek-sdk-api"))
    implementation(project(":sdk:gollek-sdk-core"))
    implementation("tech.kayys.alkhawarizm:alkhawarizm-spi-model:0.1.0-SNAPSHOT")
    implementation("tech.kayys.alkhawarizm:alkhawarizm-core:0.1.0-SNAPSHOT")
    implementation("io.quarkus:quarkus-micrometer-registry-prometheus")
    
    testImplementation(platform("org.junit:junit-bom:5.10.0"))
    testImplementation("org.junit.jupiter:junit-jupiter")
}
