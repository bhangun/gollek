/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

package tech.kayys.gollek.sdk.exception;

/**
 * Exception thrown when no plugins are available at all.
 *
 * <p>This exception indicates that the Gollek platform has no plugins installed
 * and provides comprehensive installation instructions.</p>
 *
 * @since 2.0.0
 */
public class NoPluginsAvailableException extends SdkException {

    private final String pluginDirectory;
    private final String deploymentMode;

    /**
     * Create no plugins available exception.
     *
     * @param pluginDirectory plugin directory path
     * @param deploymentMode deployment mode (standalone, microservice, hybrid)
     */
    public NoPluginsAvailableException(String pluginDirectory, String deploymentMode) {
        super(buildMessage(pluginDirectory, deploymentMode));
        this.pluginDirectory = pluginDirectory;
        this.deploymentMode = deploymentMode;
    }

    /**
     * Create no plugins available exception with cause.
     *
     * @param pluginDirectory plugin directory path
     * @param deploymentMode deployment mode
     * @param cause root cause
     */
    public NoPluginsAvailableException(String pluginDirectory, String deploymentMode, Throwable cause) {
        super(buildMessage(pluginDirectory, deploymentMode), cause);
        this.pluginDirectory = pluginDirectory;
        this.deploymentMode = deploymentMode;
    }

    /**
     * Get plugin directory.
     *
     * @return plugin directory path
     */
    public String getPluginDirectory() {
        return pluginDirectory;
    }

    /**
     * Get deployment mode.
     *
     * @return deployment mode
     */
    public String getDeploymentMode() {
        return deploymentMode;
    }

    /**
     * Get installation instructions.
     *
     * @return comprehensive installation instructions
     */
    public String getInstallationInstructions() {
        StringBuilder instructions = new StringBuilder();
        
        if ("standalone".equalsIgnoreCase(deploymentMode)) {
            instructions.append("\n\n📦 STANDALONE MODE: No plugins were included in this build.\n\n");
            instructions.append("   To add plugins:\n");
            instructions.append("   1. Rebuild with plugins: mvn clean install -Pinclude-plugins\n");
            instructions.append("   2. Or switch to microservice mode and install plugins dynamically\n");
        } else {
            instructions.append("\n\n📦 INSTALLATION OPTIONS:\n\n");
            instructions.append("   Option 1: Download Individual Plugins\n");
            instructions.append("   ─────────────────────────────────────\n");
            instructions.append("   1. Visit: https://gollek.ai/plugins\n");
            instructions.append("   2. Download the plugins you need\n");
            instructions.append("   3. Place plugin JARs in: ").append(pluginDirectory).append("\n");
            instructions.append("   4. Restart the application\n\n");
            
            instructions.append("   Option 2: Use Package Manager\n");
            instructions.append("   ─────────────────────────────────\n");
            instructions.append("   gollek install --all\n");
            instructions.append("   gollek install <plugin-id>\n\n");
            
            instructions.append("   Option 3: Build from Source\n");
            instructions.append("   ──────────────────────────────\n");
            instructions.append("   cd inference-gollek\n");
            instructions.append("   mvn clean install -DskipTests\n");
            instructions.append("   cp plugins/*/target/*.jar ").append(pluginDirectory).append("\n");
        }
        
        return instructions.toString();
    }

    private static String buildMessage(String pluginDirectory, String deploymentMode) {
        return String.format(
            "No plugins available. Deployment mode: %s. Plugin directory: %s",
            deploymentMode,
            pluginDirectory
        );
    }
}
