package tech.kayys.golek.core.plugin;

import java.net.URL;

/**
 * Plugin classloader for isolation
 */
public class PluginClassLoader extends ClassLoader {
    private final String pluginId;
    private final URL[] urls;

    public PluginClassLoader(String pluginId, URL[] urls, ClassLoader parent) {
        super(parent);
        this.pluginId = pluginId;
        this.urls = urls != null ? urls.clone() : new URL[0];
    }

    @Override
    protected Class<?> findClass(String name) throws ClassNotFoundException {
        // Custom class loading logic for plugin isolation
        // This would load classes from the plugin's JAR files
        return super.findClass(name);
    }

    @Override
    protected URL findResource(String name) {
        // Custom resource loading logic for plugin isolation
        return super.findResource(name);
    }

    public String getPluginId() {
        return pluginId;
    }

    public URL[] getUrls() {
        return urls.clone();
    }

    @Override
    public String toString() {
        return "PluginClassLoader{" +
                "pluginId='" + pluginId + '\'' +
                ", urls=" + java.util.Arrays.toString(urls) +
                '}';
    }
}