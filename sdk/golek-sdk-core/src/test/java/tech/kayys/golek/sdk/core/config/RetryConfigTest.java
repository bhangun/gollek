package tech.kayys.golek.sdk.core.config;

import org.junit.jupiter.api.Test;

import java.time.Duration;

import static org.junit.jupiter.api.Assertions.*;

class RetryConfigTest {

    @Test
    void testDefaultConfiguration() {
        RetryConfig config = RetryConfig.builder().build();
        
        assertEquals(3, config.getMaxAttempts());
        assertEquals(Duration.ofMillis(100), config.getInitialBackoff());
        assertEquals(Duration.ofSeconds(30), config.getMaxBackoff());
        assertEquals(2.0, config.getBackoffMultiplier());
        assertTrue(config.isEnableJitter());
    }

    @Test
    void testCustomConfiguration() {
        RetryConfig config = RetryConfig.builder()
                .maxAttempts(5)
                .initialBackoff(Duration.ofMillis(200))
                .maxBackoff(Duration.ofMinutes(1))
                .backoffMultiplier(3.0)
                .enableJitter(false)
                .build();
        
        assertEquals(5, config.getMaxAttempts());
        assertEquals(Duration.ofMillis(200), config.getInitialBackoff());
        assertEquals(Duration.ofMinutes(1), config.getMaxBackoff());
        assertEquals(3.0, config.getBackoffMultiplier());
        assertFalse(config.isEnableJitter());
    }

    @Test
    void testBackoffCalculation() {
        RetryConfig config = RetryConfig.builder()
                .initialBackoff(Duration.ofMillis(100))
                .backoffMultiplier(2.0)
                .maxBackoff(Duration.ofSeconds(10))
                .enableJitter(false)
                .build();
        
        // First attempt: 100ms
        assertEquals(Duration.ofMillis(100), config.calculateBackoff(1));
        
        // Second attempt: 200ms
        assertEquals(Duration.ofMillis(200), config.calculateBackoff(2));
        
        // Third attempt: 400ms
        assertEquals(Duration.ofMillis(400), config.calculateBackoff(3));
        
        // Should cap at maxBackoff
        assertTrue(config.calculateBackoff(10).toMillis() <= Duration.ofSeconds(10).toMillis());
    }

    @Test
    void testBackoffWithJitter() {
        RetryConfig config = RetryConfig.builder()
                .initialBackoff(Duration.ofMillis(100))
                .backoffMultiplier(2.0)
                .enableJitter(true)
                .build();
        
        Duration backoff1 = config.calculateBackoff(1);
        Duration backoff2 = config.calculateBackoff(1);
        
        // With jitter, two calculations should likely be different
        // (though there's a small chance they could be equal)
        assertTrue(backoff1.toMillis() >= 100);
        assertTrue(backoff2.toMillis() >= 100);
    }

    @Test
    void testInvalidMaxAttempts() {
        assertThrows(IllegalArgumentException.class, () -> 
            RetryConfig.builder().maxAttempts(0).build()
        );
    }
}
