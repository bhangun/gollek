plugins {
    java
}

val quarkusVersion = rootProject.extra["quarkusVersion"] as String

dependencies {
    implementation(project(":core:gollek-runtime-config"))
    implementation("tech.kayys.aljabr:aljabr-tensor:0.1.0-SNAPSHOT")
    implementation(project(":spi:gollek-spi-inference"))
    implementation(project(":spi:gollek-spi-model"))
    implementation(project(":spi:gollek-spi"))
    implementation(project(":core:gollek-error-code"))
    implementation("com.fasterxml.jackson.core:jackson-databind:2.16.1")
    implementation("com.fasterxml.jackson.datatype:jackson-datatype-jsr310:2.16.1")
    implementation("io.smallrye.reactive:mutiny:2.5.5")
    implementation("jakarta.enterprise:jakarta.enterprise.cdi-api:4.0.1")
    implementation("io.quarkus:quarkus-cache:$quarkusVersion")
    implementation("jakarta.ws.rs:jakarta.ws.rs-api:3.1.0")
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
