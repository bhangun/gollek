package tech.kayys.gollek.client;

import tech.kayys.gollek.sdk.config.SdkConfig;
import tech.kayys.gollek.sdk.core.GollekSdk;
import tech.kayys.gollek.sdk.core.GollekSdkProvider;
import tech.kayys.gollek.sdk.exception.SdkException;

/**
 * ServiceLoader-registered provider for the remote (HTTP-based) SDK
 * implementation.
 *
 * <p>
 * This provider creates {@link GollekClient} instances that communicate with
 * the inference engine via HTTP API. It is automatically discovered by
 * {@code GollekSdkFactory} when {@code gollek-sdk-java-remote} is on the
 * classpath.
 *
 * <p>
 * Developers can also use {@link GollekClient#builder()} directly
 * without going through the factory.
 */
public class RemoteGollekSdkProvider implements GollekSdkProvider {

    @Override
    public Mode mode() {
        return Mode.REMOTE;
    }

    @Override
    public GollekSdk create(SdkConfig config) throws SdkException {
        if (config == null) {
            throw new SdkException("SDK_ERR_CONFIG", "SdkConfig is required for remote SDK");
        }

        GollekClient.Builder builder = GollekClient.builder()
                .apiKey(config.getApiKey())
                .connectTimeout(config.getConnectTimeout());

        config.getPreferredProvider().ifPresent(builder::preferredProvider);

        return builder.build();
    }

    @Override
    public int priority() {
        return 100; // Lower priority than local
    }
}
