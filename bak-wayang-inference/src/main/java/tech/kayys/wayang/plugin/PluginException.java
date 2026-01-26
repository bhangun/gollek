package tech.kayys.golek.plugin;


public class PluginException extends Exception {
    public PluginException(String message) {
        super(message);
    }
    
    public PluginException(String message, Throwable cause) {
        super(message, cause);
    }
}
