plugins {
    java
}

// Read Feature Flags from gradle.properties or command line
val backends: List<String> = project.findProperty("gollek.backend")?.toString()?.split(",") ?: listOf("cpu")

val enableInference: Boolean = project.findProperty("gollek.inference")?.toString()?.toBoolean() ?: true

println("⚙️ Configuring gollek-core build:")
println("   - Backends: $backends")
println("   - Inference Enabled: $enableInference")

val quarkusVersion = rootProject.extra["quarkusVersion"] as String

dependencies {
    implementation(platform("io.quarkus.platform:quarkus-bom:$quarkusVersion"))
    implementation("io.quarkus:quarkus-arc")
    implementation("io.smallrye.config:smallrye-config:3.10.1")
    implementation("tech.kayys.alkhawarizm:alkhawarizm-tensor:0.1.0-SNAPSHOT")
    // Optional intra-build project dependencies — only add if those projects are present
    val spiGollek = findProject(":spi:gollek-spi")
    if (spiGollek != null) {
        add("implementation", spiGollek)
    } else {
        println("   [Warning] spi:gollek-spi not present; skipping project dependency")
    }
    val spiInference = findProject(":spi:gollek-spi-inference")
    if (spiInference != null) {
        add("implementation", spiInference)
    } else {
        println("   [Warning] spi:gollek-spi-inference not present; skipping project dependency")
    }
    val spiModel = findProject(":spi:gollek-spi-model")
    if (spiModel != null) {
        add("implementation", spiModel)
    } else {
        println("   [Warning] spi:gollek-spi-model not present; skipping project dependency")
    }
    // Depend on published alkhawarizm core aggregator for shared math/tensor/tokenizer foundations
    implementation("tech.kayys.alkhawarizm:alkhawarizm-core:0.1.0-SNAPSHOT")
    
    implementation("io.smallrye.reactive:mutiny:2.5.5")
    implementation("jakarta.enterprise:jakarta.enterprise.cdi-api:4.0.1")

    testImplementation("org.junit.jupiter:junit-jupiter:5.11.2")
}

tasks.test {
    useJUnitPlatform()
}

// Dynamically include/exclude packages based on feature flags
sourceSets {
    main {
        java {


            // Exclude inference/runner modules if not needed
            if (!enableInference) {
                println("   [Optimizer] Excluded 'runtime' and 'diffusion' packages.")
                exclude("tech/kayys/gollek/runtime/**")
                exclude("tech/kayys/gollek/diffusion/**")
            }

            // Exclude backend implementations that are not requested
            if (!backends.contains("cuda")) {
                println("   [Optimizer] Excluded 'backend/cuda' package.")
                exclude("tech/kayys/gollek/backend/cuda/**")
            }
            if (!backends.contains("metal")) {
                println("   [Optimizer] Excluded 'backend/metal' package.")
                exclude("tech/kayys/gollek/backend/metal/**")
            }
            if (!backends.contains("cpu")) {
                println("   [Optimizer] Excluded 'backend/cpu' package.")
                exclude("tech/kayys/gollek/backend/cpu/**")
            }
        }
    }
}
