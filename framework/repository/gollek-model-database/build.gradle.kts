plugins {
    java
}

dependencies {
    implementation("tech.kayys.alkhawarizm:alkhawarizm-spi-model:0.1.0-SNAPSHOT")
    implementation("io.quarkus:quarkus-hibernate-reactive-panache")
    implementation("io.quarkus:quarkus-reactive-pg-client")
    testImplementation(platform("org.junit:junit-bom:5.10.0"))
    testImplementation("org.junit.jupiter:junit-jupiter")
}

tasks.test {
    useJUnitPlatform()
}
