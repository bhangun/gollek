package tech.kayys.golek.core;

import io.quarkus.runtime.annotations.RegisterForReflection;
import tech.kayys.golek.api.AuditPayload;
import tech.kayys.golek.api.ErrorPayload;
import tech.kayys.golek.api.Message;
import tech.kayys.golek.api.inference.InferenceRequest;
import tech.kayys.golek.api.inference.InferenceResponse;

// Commented out - requires GraalVM native-image dependencies
// import org.graalvm.nativeimage.hosted.Feature;

/**
 * GraalVM native image configuration.
 * Note: GraalVM Feature interface commented out - requires native-image dependencies
 */
@RegisterForReflection(targets = {
        InferenceRequest.class,
        InferenceResponse.class,
        Message.class,
        ErrorPayload.class,
        AuditPayload.class
})
public class NativeImageFeature /* implements Feature */ {

    // Commented out - requires GraalVM native-image dependencies
    /*
    @Override
    public void beforeAnalysis(BeforeAnalysisAccess access) {
        // Register classes for reflection
        access.registerAsInHeap(InferenceRequest.class);
        access.registerAsInHeap(InferenceResponse.class);
    }
    */
}