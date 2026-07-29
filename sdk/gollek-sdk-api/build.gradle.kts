plugins {
    java
}

dependencies {
    implementation("tech.kayys.alkhawarizm:alkhawarizm-tensor:0.1.0-SNAPSHOT")
    implementation(project(":spi:gollek-spi-inference"))
    implementation(project(":spi:gollek-spi-model"))
    implementation(project(":spi:gollek-spi-multimodal"))
    implementation(project(":runner:safetensor:gollek-safetensor-core"))
    implementation(project(":runner:safetensor:gollek-safetensor-quantization"))
    implementation("io.smallrye.reactive:mutiny:2.5.5")
    implementation("jakarta.enterprise:jakarta.enterprise.cdi-api:4.0.1")
    implementation("jakarta.inject:jakarta.inject-api:2.0.1")
    implementation("org.jboss.logging:jboss-logging:3.5.3.Final")
}
