plugins {
    java
}

dependencies {
    implementation(project(":core:gollek-core"))
    implementation(project(":spi:gollek-spi"))
    implementation(project(":spi:gollek-spi-runner"))
    implementation(project(":spi:gollek-spi-plugin"))
    implementation(project(":spi:gollek-spi-inference"))
    implementation(project(":runner:gguf:gollek-runner-gguf"))
    implementation(project(":plugin:gollek-plugin-runner-core"))
    
    // Add alkhawarizm model spi for MultimodalRequest
    implementation("tech.kayys.alkhawarizm:alkhawarizm-spi-model:0.1.0-SNAPSHOT")
    
    // Add other typical dependencies
    implementation("org.jboss.logging:jboss-logging")
    implementation("io.smallrye.reactive:mutiny:2.5.5")
    implementation("io.micrometer:micrometer-core:1.14.4")
    
    // Jakarta / MicroProfile dependencies
    implementation("jakarta.enterprise:jakarta.enterprise.cdi-api:4.0.1")
    implementation("jakarta.inject:jakarta.inject-api:2.0.1")
    implementation("jakarta.annotation:jakarta.annotation-api:2.1.1")
    
    testImplementation(project(":sdk:gollek-sdk-core"))
}
