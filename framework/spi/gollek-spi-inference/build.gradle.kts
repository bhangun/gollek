plugins {
    java
}

dependencies {
    implementation(project(":spi:gollek-spi"))
    implementation(project(":core:gollek-error-code"))
    implementation("tech.kayys.alkhawarizm:alkhawarizm-spi-model:0.1.0-SNAPSHOT")
    implementation(project(":spi:gollek-spi-multimodal"))
    implementation("io.smallrye.reactive:mutiny:2.5.5")
    implementation("org.reactivestreams:reactive-streams:1.0.4")
    implementation("com.fasterxml.jackson.core:jackson-annotations:2.16.1")
    implementation("jakarta.validation:jakarta.validation-api:3.0.2")
    implementation("org.jetbrains:annotations:24.0.1")
    
    testImplementation(group = "org.junit.jupiter", name = "junit-jupiter")
}

tasks.withType<Test>().configureEach {
    useJUnitPlatform()
}
