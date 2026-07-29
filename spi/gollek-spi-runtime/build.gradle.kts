plugins {
    java
}

dependencies {
    implementation(project(":spi:gollek-spi"))
    implementation(project(":spi:gollek-spi-inference"))
    implementation(project(":spi:gollek-spi-model"))
    implementation("tech.kayys.alkhawarizm:alkhawarizm-tensor:0.1.0-SNAPSHOT")
    implementation("io.smallrye.reactive:mutiny:2.5.5")
    implementation("jakarta.enterprise:jakarta.enterprise.cdi-api:4.0.1")
    implementation("jakarta.inject:jakarta.inject-api:2.0.1")
    implementation("org.jboss.logging:jboss-logging:3.5.3.Final")
}
