import org.jetbrains.compose.desktop.application.dsl.TargetFormat

plugins {
    kotlin("jvm")
    id("org.jetbrains.compose") version "1.5.11"
}

dependencies {
    implementation(project(":shared"))
    implementation(compose.desktop.currentOs)
    implementation(compose.material3)
    implementation(compose.materialIconsExtended)
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-swing:1.7.3")
}

compose.desktop {
    application {
        mainClass = "com.kmpchatbot.desktop.MainKt"
        
        nativeDistributions {
            targetFormats(TargetFormat.Dmg, TargetFormat.Msi, TargetFormat.Deb)
            packageName = "KMPChatbot"
            packageVersion = "1.0.0"
            
            windows {
                iconFile.set(project.file("icon.ico"))
            }
            macOS {
                iconFile.set(project.file("icon.icns"))
            }
            linux {
                iconFile.set(project.file("icon.png"))
            }
        }
    }
}
