plugins {
    `java-library`
}

dependencies {
    implementation("tech.kayys.alkhawarizm:alkhawarizm-tensor:0.1.0-SNAPSHOT")
    api("com.fasterxml.jackson.core:jackson-annotations:2.16.1")
    api("io.smallrye.reactive:mutiny:2.5.5")
    implementation("com.fasterxml.jackson.core:jackson-databind:2.16.1")
    implementation("org.jboss.logging:jboss-logging:3.5.3.Final")
    testImplementation("org.junit.jupiter:junit-jupiter-api:5.10.2")
    testRuntimeOnly("org.junit.jupiter:junit-jupiter-engine:5.10.2")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher:1.10.2")
}

tasks.withType<Test>().configureEach {
    useJUnitPlatform()
}
