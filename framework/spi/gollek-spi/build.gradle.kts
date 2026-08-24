plugins {
    java
}

dependencies {
    implementation("tech.kayys.alkhawarizm:alkhawarizm-error-code:0.1.0-SNAPSHOT")
    implementation("com.fasterxml.jackson.core:jackson-annotations:2.16.1")
    implementation("jakarta.validation:jakarta.validation-api:3.0.2")
    implementation("commons-codec:commons-codec:1.16.1")
    implementation("org.jetbrains:annotations:24.0.1")
    implementation("io.smallrye.reactive:mutiny:2.5.5")
    implementation("jakarta.ws.rs:jakarta.ws.rs-api:3.1.0")
    implementation("tech.kayys.alkhawarizm:alkhawarizm-tensor:0.1.0-SNAPSHOT")

    testImplementation(platform("org.junit:junit-bom:5.10.0"))
    testImplementation("org.junit.jupiter:junit-jupiter")
    testImplementation("io.quarkus:quarkus-junit")
}

tasks.withType<Test>().configureEach {
    useJUnitPlatform()
}
