plugins {
    java
}

dependencies {
    implementation(project(":sdk:gollek-sdk-core"))
    implementation(project(":sdk:gollek-sdk-api"))
    implementation(project(":sdk:gollek-sdk-protobuf"))
    implementation("io.grpc:grpc-protobuf:1.64.0")
    implementation("io.grpc:grpc-stub:1.64.0")
    implementation("io.grpc:grpc-netty-shaded:1.64.0")
    implementation("tech.kayys.alkhawarizm:alkhawarizm-spi-model:0.1.0-SNAPSHOT")
    implementation(project(":spi:gollek-spi-inference"))
    implementation("io.smallrye.reactive:mutiny:2.5.5")
    implementation("com.fasterxml.jackson.core:jackson-databind:2.16.1")
}
