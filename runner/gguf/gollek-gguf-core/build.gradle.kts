plugins {
    `java-library`
}

sourceSets {
    main {
        java {
            // Keep the actively used GGUF metadata/parser/tokenizer surface.
            // The module also contains older exporter/runner/training trees that
            // do not match today's package layout and pull in stale dependencies.
            include(
                "tech/kayys/gollek/gguf/core/GGUFConstants.java",
                "tech/kayys/gollek/gguf/core/GGUFTensorInfo.java",
                "tech/kayys/gollek/gguf/core/GgmlType.java",
                "tech/kayys/gollek/gguf/core/GgufMetaType.java",
                "tech/kayys/gollek/gguf/core/GgufMetaValue.java",
                "tech/kayys/gollek/gguf/core/GgufExporter.java",
                "tech/kayys/gollek/gguf/core/GgufModel.java",
                "tech/kayys/gollek/gguf/loader/GGUFModel.java",
                "tech/kayys/gollek/gguf/loader/GGUFTensorInfo.java",
                "tech/kayys/gollek/gguf/loader/GGUFParser.java",
                "tech/kayys/gollek/gguf/loader/GGUFReader.java",
                "tech/kayys/gollek/gguf/loader/gguf/*.java",
                "tech/kayys/gollek/gguf/loader/quant/*.java",
                "tech/kayys/gollek/gguf/runtime/*.java",
                "tech/kayys/gollek/gguf/tokenizer/GGUFTokenizer.java",
                "tech/kayys/gollek/gguf/writer/GGUFWriter.java",
                "tech/kayys/gollek/gguf/model/alkhawarizm/*.java",
                "tech/kayys/gollek/gguf/model/ModelConfig.java",
                "tech/kayys/gollek/gguf/runner/AljabrWeightAdapter.java",
            )
        }
    }
}

dependencies {
    api("tech.kayys.alkhawarizm:alkhawarizm-nn:0.1.0")
    api("tech.kayys.alkhawarizm:alkhawarizm-tensor:0.1.0")
    implementation(project(":spi:gollek-spi-model"))
    implementation("com.google.code.gson:gson:2.10.1")
    implementation(project(":core:gollek-tokenizer-core"))
    implementation("io.quarkus:quarkus-core:3.32.2")
    implementation("org.jboss.logging:jboss-logging:3.6.1.Final")
    testImplementation("org.junit.jupiter:junit-jupiter")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
}

tasks.withType<Test>().configureEach {
    useJUnitPlatform()
}
