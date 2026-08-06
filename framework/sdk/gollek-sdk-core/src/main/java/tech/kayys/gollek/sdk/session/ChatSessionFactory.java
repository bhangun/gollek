package tech.kayys.gollek.sdk.session;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.inject.Instance;
import jakarta.enterprise.inject.spi.CDI;
import tech.kayys.gollek.sdk.core.GollekSdk;

/**
 * CDI-aware factory for creating {@link ChatSession} instances.
 *
 * <p>The factory can be discovered in modules that do not provide a concrete
 * {@link GollekSdk} bean. It resolves the SDK lazily so unrelated extensions can
 * boot, while session creation still fails clearly if no SDK is available.
 */
@ApplicationScoped
public class ChatSessionFactory {

    public ChatSession createSession(String modelId, String providerId) {
        return createSession(modelId, providerId, true);
    }

    public ChatSession createSession(String modelId, String providerId, boolean enableSession) {
        return new ChatSessionImpl(resolveSdk(), modelId, providerId, enableSession);
    }

    private GollekSdk resolveSdk() {
        Instance<GollekSdk> candidates = CDI.current().select(GollekSdk.class);
        if (candidates.isUnsatisfied()) {
            throw new IllegalStateException("No GollekSdk CDI bean is available for chat session creation.");
        }
        if (candidates.isAmbiguous()) {
            throw new IllegalStateException("Multiple GollekSdk CDI beans are available for chat session creation.");
        }
        return candidates.get();
    }
}
