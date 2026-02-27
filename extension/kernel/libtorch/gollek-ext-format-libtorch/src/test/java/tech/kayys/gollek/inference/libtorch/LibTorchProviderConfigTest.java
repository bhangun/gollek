package tech.kayys.gollek.inference.libtorch;

import io.quarkus.test.junit.QuarkusTest;
import jakarta.inject.Inject;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;

import static org.assertj.core.api.Assertions.assertThat;

@QuarkusTest
class LibTorchProviderConfigTest {

    @Inject
    LibTorchProviderConfig config;

    @Test
    @DisplayName("Config should load default values")
    void testDefaultValues() {
        assertThat(config.enabled()).isTrue();
        assertThat(config.model().basePath()).contains(".gollek/models/torchscript");
        assertThat(config.model().extensions()).contains(".pt");

        assertThat(config.gpu().enabled()).isFalse();
        assertThat(config.gpu().deviceIndex()).isEqualTo(0);

        assertThat(config.session().maxPerTenant()).isEqualTo(4);
        assertThat(config.session().idleTimeoutSeconds()).isEqualTo(300);
        assertThat(config.session().maxTotal()).isEqualTo(16);

        assertThat(config.inference().timeoutSeconds()).isEqualTo(30);
        assertThat(config.inference().threads()).isEqualTo(4);
    }

    @Test
    @DisplayName("Config values should be valid")
    void testValidValues() {
        assertThat(config.session().maxPerTenant()).isPositive();
        assertThat(config.session().maxTotal()).isGreaterThanOrEqualTo(config.session().maxPerTenant());
        assertThat(config.inference().timeoutSeconds()).isPositive();
        assertThat(config.inference().threads()).isPositive();
    }
}
