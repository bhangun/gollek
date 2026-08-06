plugins {
    java
}

dependencies {
    implementation("tech.kayys.alkhawarizm:alkhawarizm-tensor:0.1.0-SNAPSHOT")
    implementation(project(":core:gollek-ir"))
    implementation(project(":core:gollek-core"))
    implementation("com.fasterxml.jackson.core:jackson-databind:2.16.1")
}
