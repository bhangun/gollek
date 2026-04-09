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
 * Exception thrown when a required plugin is not available.
 *
 * <p>This exception provides detailed information about the missing plugin
 * and instructions on how to install it.</p>
 *
 * @since 2.0.0
 */
public class PluginNotAvailableException extends SdkException {

    private final String pluginType;
    private final String pluginId;
    private final String pluginDirectory;

    /**
     * Create plugin not available exception.
     *
     * @param pluginType type of plugin (provider, runner, kernel, optimization)
     * @param pluginId plugin ID
     * @param pluginDirectory plugin directory path
     */
    public PluginNotAvailableException(String pluginType, String pluginId, String pluginDirectory) {
        super(buildMessage(pluginType, pluginId, pluginDirectory));
        this.pluginType = pluginType;
        this.pluginId = pluginId;
        this.pluginDirectory = pluginDirectory;
    }

    /**
     * Create plugin not available exception with cause.
     *
     * @param pluginType type of plugin
     * @param pluginId plugin ID
     * @param pluginDirectory plugin directory path
     * @param cause root cause
     */
    public PluginNotAvailableException(String pluginType, String pluginId, String pluginDirectory, Throwable cause) {
        super(buildMessage(pluginType, pluginId, pluginDirectory), cause);
        this.pluginType = pluginType;
        this.pluginId = pluginId;
        this.pluginDirectory = pluginDirectory;
    }

    /**
     * Get plugin type.
     *
     * @return plugin type
     */
    public String getPluginType() {
        return pluginType;
    }

    /**
     * Get plugin ID.
     *
     * @return plugin ID
     */
    public String getPluginId() {
        return pluginId;
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
     * Get installation instructions.
     *
     * @return installation instructions
     */
    public String getInstallationInstructions() {
        StringBuilder instructions = new StringBuilder();
        instructions.append("\n\n📦 To install this ").append(pluginType).append(":\n");
        instructions.append("   1. Download from: https://gollek.ai/plugins\n");
        instructions.append("   2. Place in: ").append(pluginDirectory).append("/").append(pluginType).append("s/\n");
        instructions.append("   3. Restart the application\n");
        return instructions.toString();
    }

    private static String buildMessage(String pluginType, String pluginId, String pluginDirectory) {
        return String.format(
            "%s plugin '%s' is not available. Plugin directory: %s",
            capitalize(pluginType),
            pluginId,
            pluginDirectory
        );
    }

    private static String capitalize(String str) {
        if (str == null || str.isEmpty()) {
            return str;
        }
        return Character.toUpperCase(str.charAt(0)) + str.substring(1);
    }
}
