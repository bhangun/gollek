package tech.kayys.gollek.engine;

import io.quarkus.runtime.annotations.RegisterForReflection;
import tech.kayys.gollek.error.ErrorPayload;
import tech.kayys.gollek.spi.Message;
import tech.kayys.gollek.spi.context.RequestContext;
import tech.kayys.gollek.spi.execution.ExecutionContext;
import tech.kayys.gollek.spi.execution.ExecutionToken;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.model.DeviceType;
import tech.kayys.gollek.spi.observability.AuditPayload;
import tech.kayys.gollek.spi.plugin.PluginState;
import tech.kayys.gollek.spi.provider.ProviderConfig;
import tech.kayys.gollek.spi.provider.ProviderRequest;
import tech.kayys.gollek.spi.provider.ProviderResponse;

import org.graalvm.nativeimage.hosted.Feature;
import org.graalvm.nativeimage.hosted.RuntimeClassInitialization;
import org.graalvm.nativeimage.hosted.RuntimeReflection;

import java.util.Arrays;
import java.util.List;

/**
 * GraalVM native image configuration for Gollek inference engine.
 * 
 * <p>This feature registers classes for reflection, resource bundles, and
 * initializes classes at build time to ensure proper native image generation.</p>
 * 
 * <h2>Registered Components</h2>
 * <ul>
 *   <li>SPI interfaces and implementations</li>
 *   <li>Mutiny reactive types</li>
 *   <li>Jackson serialization classes</li>
 *   <li>Plugin system classes</li>
 *   <li>Provider request/response classes</li>
 * </ul>
 * 
 * <h2>Building Native Image</h2>
 * <pre>
 * # Build native image
 * mvn package -Pnative
 * 
 * # Build with container (recommended)
 * mvn package -Pnative -Dquarkus.native.container-build=true
 * 
 * # Build with specific builder image
 * mvn package -Pnative \
 *   -Dquarkus.native.container-build=true \
 *   -Dquarkus.native.builder-image=quay.io/quarkus/ubi-quarkus-mandrel-builder-image:jdk-21
 * </pre>
 * 
 * @since 1.0.0
 */
@RegisterForReflection(targets = {
    // Core SPI classes
    InferenceRequest.class,
    InferenceResponse.class,
    Message.class,
    
    // Context classes
    RequestContext.class,
    ExecutionContext.class,
    ExecutionToken.class,
    
    // Provider classes
    ProviderConfig.class,
    ProviderRequest.class,
    ProviderResponse.class,
    
    // Error and observability
    ErrorPayload.class,
    AuditPayload.class,
    
    // Plugin system
    PluginState.class,
    
    // Model classes
    DeviceType.class
}, classNames = {
    // Additional classes to register by name
    "io.smallrye.mutiny.Uni",
    "io.smallrye.mutiny.Multi",
    "io.smallrye.mutiny.subscription.UniSubscriber",
    "io.smallrye.mutiny.subscription.MultiSubscriber",
    
    // Jackson classes
    "com.fasterxml.jackson.databind.ObjectMapper",
    "com.fasterxml.jackson.databind.JsonNode",
    "com.fasterxml.jackson.databind.node.ObjectNode",
    "com.fasterxml.jackson.databind.node.ArrayNode",
    
    // Vert.x classes
    "io.vertx.core.json.JsonObject",
    "io.vertx.core.json.JsonArray"
})
public class NativeImageFeature implements Feature {

    /**
     * List of packages to register for reflection.
     */
    private static final List<String> REFLECTION_PACKAGES = Arrays.asList(
        "tech.kayys.gollek.spi.",
        "tech.kayys.gollek.error.",
        "tech.kayys.gollek.provider.core.",
        "io.smallrye.mutiny.",
        "com.fasterxml.jackson.databind.",
        "jakarta.enterprise.",
        "org.eclipse.microprofile.config."
    );

    /**
     * Classes to initialize at run time (not build time).
     */
    private static final List<String> RUNTIME_INIT_CLASSES = Arrays.asList(
        "java.net.InetAddress",
        "java.net.InetSocketAddress",
        "java.net.URL",
        "javax.net.ssl.SSLContext",
        "sun.security.ssl.SSLContextImpl",
        "io.netty.channel.epoll.Epoll",
        "io.netty.channel.epoll.EpollEventLoopGroup",
        "io.netty.channel.kqueue.KQueue",
        "io.netty.channel.kqueue.KQueueEventLoopGroup"
    );

    @Override
    public void beforeAnalysis(BeforeAnalysisAccess access) {
        // Register SPI interfaces and implementations for reflection
        registerSPIClasses(access);
        
        // Register Mutiny types
        registerMutinyTypes(access);
        
        // Register Jackson types
        registerJacksonTypes(access);
        
        // Register resource bundles
        registerResourceBundles(access);
    }

    @Override
    public void duringAnalysis(DuringAnalysisAccess access) {
        // Additional registration during analysis phase
        registerAdditionalResources(access);
    }

    /**
     * Register SPI classes for reflection.
     */
    private void registerSPIClasses(BeforeAnalysisAccess access) {
        // Register specific SPI classes
        registerClass(access, "tech.kayys.gollek.spi.inference.InferencePhasePlugin");
        registerClass(access, "tech.kayys.gollek.spi.plugin.GollekPlugin");
        registerClass(access, "tech.kayys.gollek.spi.plugin.GollekConfigurablePlugin");
    }

    /**
     * Register Mutiny reactive types.
     */
    private void registerMutinyTypes(BeforeAnalysisAccess access) {
        registerClass(access, "io.smallrye.mutiny.Uni");
        registerClass(access, "io.smallrye.mutiny.Multi");
        registerClass(access, "io.smallrye.mutiny.groups.UniSubscribe");
        registerClass(access, "io.smallrye.mutiny.groups.MultiSubscribe");
    }

    /**
     * Register Jackson serialization types.
     */
    private void registerJacksonTypes(BeforeAnalysisAccess access) {
        registerClass(access, "com.fasterxml.jackson.databind.Module");
        registerClass(access, "com.fasterxml.jackson.databind.ser.Serializers");
        registerClass(access, "com.fasterxml.jackson.databind.deser.Deserializers");
    }

    /**
     * Register resource bundles needed at runtime.
     */
    private void registerResourceBundles(BeforeAnalysisAccess access) {
        // Register any resource bundles if needed
        // Example: RuntimeReflection.registerResourceBundle("messages");
    }

    /**
     * Register additional resources (files, etc.).
     */
    private void registerAdditionalResources(DuringAnalysisAccess access) {
        // Register configuration files
        // Example: RuntimeReflection.registerResource("application.properties");
    }

    /**
     * Helper method to register a class by name.
     */
    private void registerClass(BeforeAnalysisAccess access, String className) {
        try {
            Class<?> clazz = access.findClassByName(className);
            if (clazz != null) {
                RuntimeReflection.register(clazz);
                RuntimeReflection.register(clazz.getDeclaredMethods());
                RuntimeReflection.register(clazz.getDeclaredFields());
                RuntimeReflection.register(clazz.getDeclaredConstructors());
            }
        } catch (Exception e) {
            // Class might not be on classpath, continue
        }
    }

    /**
     * Initialize classes at run time.
     */
    private void initializeAtRuntime() {
        RUNTIME_INIT_CLASSES.forEach(className -> {
            try {
                Class<?> clazz = Class.forName(className);
                RuntimeClassInitialization.initializeAtRunTime(clazz);
            } catch (ClassNotFoundException e) {
                // Class might not be on classpath, continue
            }
        });
    }
}
