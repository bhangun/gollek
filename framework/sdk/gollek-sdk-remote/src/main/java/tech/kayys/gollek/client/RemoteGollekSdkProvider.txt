package tech.kayys.gollek.client;

import tech.kayys.gollek.sdk.config.SdkConfig;
import tech.kayys.gollek.sdk.core.GollekSdk;
import tech.kayys.gollek.sdk.core.GollekSdkProvider;
import tech.kayys.gollek.sdk.exception.SdkException;

/**
 * {@link java.util.ServiceLoader}-registered provider that creates remote
 * (HTTP-based) {@link GollekSdk} instances.
 *
 * <p>Automatically discovered by {@code GollekSdkFactory} when
 * {@code gollek-sdk-java-remote} is on the classpath. It has a lower priority
 * ({@code 100}) than the local provider so the local engine is preferred when
 * both are available.
 *
 * <p>Developers can also bypass the factory and use {@link GollekClient#builder()}
 * directly for full control over connection settings.
 *
 * @see GollekClient
 */
public class RemoteGollekSdkProvider implements GollekSdkProvider {

    /**
     * Returns {@link Mode#REMOTE}, indicating this provider communicates over HTTP.
     *
     * @return {@link Mode#REMOTE}
     */
    @Override
    public Mode mode() {
        return Mode.REMOTE;
    }

    /**
     * Creates a {@link GollekClient} configured from the given {@link SdkConfig}.
     *
     * @param config SDK configuration; must not be {@code null} — use
     *               {@link GollekClient#builder()} directly if you need defaults
     * @return a configured remote {@link GollekSdk} instance
     * @throws SdkException with code {@code "SDK_ERR_CONFIG"} if {@code config} is {@code null}
     */
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

    /**
     * Returns {@code 100} — lower priority than the local provider ({@code 10}),
     * so local is preferred when both are on the classpath.
     *
     * @return {@code 100}
     */
    @Override
    public int priority() {
        return 100;
    }
}
