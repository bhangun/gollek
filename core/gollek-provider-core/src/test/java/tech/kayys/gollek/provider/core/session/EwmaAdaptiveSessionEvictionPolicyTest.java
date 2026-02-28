package tech.kayys.gollek.provider.core.session;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

class EwmaAdaptiveSessionEvictionPolicyTest {

    @Test
    void tightensAndRecoversFromTelemetry() {
        EwmaAdaptiveSessionEvictionPolicy policy = new EwmaAdaptiveSessionEvictionPolicy();
        AdaptiveSessionEvictionState state = new AdaptiveSessionEvictionState();

        int baseline = policy.resolveIdleTimeoutSeconds(state, 300, 0.10d);
        assertThat(baseline).isEqualTo(300);

        for (int i = 0; i < 8; i++) {
            policy.recordTelemetry(state, true, 0);
        }

        int tightened = policy.resolveIdleTimeoutSeconds(state, 300, 0.10d);
        assertThat(tightened).isLessThan(baseline);
        assertThat(policy.pressureScore(state)).isGreaterThan(0.60d);

        for (int i = 0; i < 12; i++) {
            policy.recordTelemetry(state, false, 0);
        }

        int recovered = policy.resolveIdleTimeoutSeconds(state, 300, 0.10d);
        assertThat(recovered).isGreaterThan(tightened);
        assertThat(policy.pressureScore(state)).isLessThan(0.35d);
    }
}

