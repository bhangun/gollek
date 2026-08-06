plugins {
    java
}

dependencies {
    implementation(project(":spi:gollek-spi"))
    implementation(project(":spi:gollek-spi-inference"))
    implementation("io.smallrye.reactive:mutiny:2.5.5")
    implementation("org.slf4j:slf4j-api:2.0.12")
}
