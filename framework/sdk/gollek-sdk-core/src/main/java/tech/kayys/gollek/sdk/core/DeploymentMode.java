package tech.kayys.gollek.sdk.core;

/**
 * Deployment mode for the Gollek platform.
 *
 * <ul>
 *   <li><b>STANDALONE</b> - All plugins built-in, compiled in one JAR</li>
 *   <li><b>MICROSERVICE</b> - Dynamic plugin loading from plugin directory</li>
 *   <li><b>HYBRID</b> - Built-in + dynamic plugins</li>
 * </ul>
 *
 * @since 2.0.0
 */
public enum DeploymentMode {
    /** Standalone/Local - All plugins built-in */
    STANDALONE,
    
    /** Microservice - Dynamic plugin loading */
    MICROSERVICE,
    
    /** Hybrid - Built-in + dynamic plugins */
    HYBRID
}
