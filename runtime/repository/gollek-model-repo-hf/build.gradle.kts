plugins {
    java
}

dependencies {
    implementation(project(":core:gollek-core"))
    implementation(project(":spi:gollek-spi"))
    implementation("tech.kayys.alkhawarizm:alkhawarizm-spi-model:0.1.0-SNAPSHOT")
    implementation("tech.kayys.alkhawarizm:alkhawarizm-tensor:0.1.0-SNAPSHOT")
    implementation(project(":core:gollek-core"))
    implementation(project(":core:gollek-model-repo-local"))
    implementation("io.quarkus:quarkus-arc")
    implementation("io.quarkus:quarkus-rest-client")
    implementation("io.quarkus:quarkus-rest-client-jackson")
    implementation("io.smallrye.reactive:mutiny")
    implementation("io.smallrye.config:smallrye-config:3.10.1")
    implementation("io.vertx:vertx-web-client:4.5.10")
    implementation("software.amazon.awssdk:s3:2.25.53")
    implementation("software.amazon.awssdk:s3-transfer-manager:2.25.53")
    implementation("com.fasterxml.jackson.core:jackson-databind:2.16.1")
    implementation("com.fasterxml.jackson.datatype:jackson-datatype-jsr310:2.16.1")
    implementation("org.jboss.logging:jboss-logging:3.6.1.Final")
    implementation("org.apache.commons:commons-lang3:3.17.0")
    implementation("commons-codec:commons-codec:1.16.1")
    implementation("commons-io:commons-io:2.18.0")
}

sourceSets {
    main {
        java {
        }
    }
}
