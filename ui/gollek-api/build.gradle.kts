plugins {
    java
    id("io.quarkus")
}

val quarkusVersion = rootProject.extra["quarkusVersion"] as String

configurations.configureEach {
    exclude(group = "tech.kayys.gollek", module = "gollek-safetensor-api")
}

dependencies {
    implementation("tech.kayys.alkhawarizm:alkhawarizm-tensor:0.1.0-SNAPSHOT")
    implementation(project(":core:gollek-ir"))
    implementation(project(":core:gollek-core"))
    implementation(platform("io.quarkus.platform:quarkus-bom:$quarkusVersion"))
    implementation("io.quarkus:quarkus-arc")
    implementation("io.quarkus:quarkus-rest")
    implementation("io.quarkus:quarkus-rest-jackson")
    implementation("io.quarkus:quarkus-smallrye-openapi")
    implementation("io.quarkus:quarkus-micrometer-registry-prometheus")
    implementation("io.quarkus:quarkus-mutiny")

    implementation(project(":sdk:gollek-sdk"))
    implementation(project(":sdk:gollek-sdk-agent"))
    implementation(project(":sdk:gollek-sdk-core"))
    implementation(project(":spi:gollek-spi"))
    implementation(project(":spi:gollek-spi-inference"))
    implementation(project(":spi:gollek-spi-model"))

    testImplementation(platform("org.junit:junit-bom:5.10.0"))
    testImplementation("org.junit.jupiter:junit-jupiter")
    testImplementation("io.quarkus:quarkus-junit5")
    testImplementation("io.rest-assured:rest-assured")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
}

tasks.test {
    useJUnitPlatform()
}
