plugins {
    java
}

val quarkusVersion = rootProject.extra["quarkusVersion"] as String

dependencies {
    implementation(project(":core:gollek-core"))
    implementation("tech.kayys.alkhawarizm:alkhawarizm-tensor:0.1.0-SNAPSHOT")
    implementation(project(":spi:gollek-spi-inference"))
    implementation("tech.kayys.alkhawarizm:alkhawarizm-spi-model:0.1.0-SNAPSHOT")
    implementation(project(":spi:gollek-spi"))
    implementation("tech.kayys.alkhawarizm:alkhawarizm-error-code:0.1.0-SNAPSHOT")
    implementation("com.fasterxml.jackson.core:jackson-databind:2.16.1")
    implementation("com.fasterxml.jackson.datatype:jackson-datatype-jsr310:2.16.1")
    implementation("io.smallrye.reactive:mutiny:2.5.5")
    implementation("jakarta.enterprise:jakarta.enterprise.cdi-api:4.0.1")
    implementation("io.quarkus:quarkus-cache:$quarkusVersion")
    implementation("jakarta.ws.rs:jakarta.ws.rs-api:3.1.0")
    testImplementation(project(":core:gollek-model-repo-local"))
    testImplementation(platform("org.junit:junit-bom:5.10.0"))
    testImplementation("org.junit.jupiter:junit-jupiter")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
}

sourceSets {
    main {
        java {
            // These admin/registry services still depend on a separate reactive
            // persistence stack and are outside the current CLI-focused Gradle
            // migration slice.
            exclude("tech/kayys/gollek/utils/detector/hw/HardwareConfig.java")
            exclude("tech/kayys/gollek/model/registry/ModelManagementService.java")
            exclude("tech/kayys/gollek/model/registry/ModelRegistryService.java")
        }
    }
}
