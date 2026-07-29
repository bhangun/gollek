plugins {
    java
}

dependencies {
    implementation("tech.kayys.alkhawarizm:alkhawarizm-tensor:0.1.0-SNAPSHOT")
    implementation(project(":core:gollek-ir"))
    implementation(project(":core:gollek-core"))
    implementation(project(":runtime:gollek-runtime"))
    testImplementation(platform("org.junit:junit-bom:5.10.0"))
    testImplementation("org.junit.jupiter:junit-jupiter")
}

tasks.test {
    useJUnitPlatform()
}
