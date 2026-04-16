package tech.kayys.gollek.metal.config;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

class MetalRunnerModeTest {

    @Test
    void defaultsToAutoWhenNullOrBlank() {
        assertThat(MetalRunnerMode.from(null)).isEqualTo(MetalRunnerMode.AUTO);
        assertThat(MetalRunnerMode.from("")).isEqualTo(MetalRunnerMode.AUTO);
        assertThat(MetalRunnerMode.from("   ")).isEqualTo(MetalRunnerMode.AUTO);
    }

    @Test
    void parsesKnownModes() {
        assertThat(MetalRunnerMode.from("auto")).isEqualTo(MetalRunnerMode.AUTO);
        assertThat(MetalRunnerMode.from("standard")).isEqualTo(MetalRunnerMode.STANDARD);
        assertThat(MetalRunnerMode.from("offload")).isEqualTo(MetalRunnerMode.OFFLOAD);
        assertThat(MetalRunnerMode.from("weight-offload")).isEqualTo(MetalRunnerMode.OFFLOAD);
        assertThat(MetalRunnerMode.from("metal-offload")).isEqualTo(MetalRunnerMode.OFFLOAD);
        assertThat(MetalRunnerMode.from("force")).isEqualTo(MetalRunnerMode.FORCE);
        assertThat(MetalRunnerMode.from("manual")).isEqualTo(MetalRunnerMode.FORCE);
        assertThat(MetalRunnerMode.from("forced")).isEqualTo(MetalRunnerMode.FORCE);
        assertThat(MetalRunnerMode.from("disabled")).isEqualTo(MetalRunnerMode.DISABLED);
        assertThat(MetalRunnerMode.from("off")).isEqualTo(MetalRunnerMode.DISABLED);
        assertThat(MetalRunnerMode.from("false")).isEqualTo(MetalRunnerMode.DISABLED);
    }

    @Test
    void fallsBackToAutoForUnknown() {
        assertThat(MetalRunnerMode.from("mystery")).isEqualTo(MetalRunnerMode.AUTO);
    }
}
