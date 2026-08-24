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
    implementation(project(":runner:gguf:gollek-plugin-runner-gguf"))

    // Add other typical dependencies
    implementation("org.jboss.logging:jboss-logging")
    implementation("io.smallrye.reactive:mutiny:2.5.5")
    implementation("io.micrometer:micrometer-core:1.14.4")
    
    testImplementation(project(":sdk:gollek-sdk-core"))
}
