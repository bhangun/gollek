package tech.kayys.gollek.plugin.security;

import org.junit.jupiter.api.*;
import static org.assertj.core.api.Assertions.assertThat;

/**
 * Unit tests for PIIRedactionService.
 * 
 * @since 1.0.0
 */
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class PIIRedactionServiceTest {

    private PIIRedactionService redactionService;

    @BeforeEach
    void setUp() {
        redactionService = new PIIRedactionService();
    }

    @Test
    @Order(1)
    @DisplayName("Should redact email addresses")
    void shouldRedactEmail() {
        // Given
        String text = "Contact me at john.doe@example.com or jane.smith@company.co.uk";

        // When
        String redacted = redactionService.redact(text);

        // Then
        assertThat(redacted).contains("[REDACTED_EMAIL]");
        assertThat(redacted).doesNotContain("john.doe@example.com");
        assertThat(redacted).doesNotContain("jane.smith@company.co.uk");
    }

    @Test
    @Order(2)
    @DisplayName("Should redact phone numbers")
    void shouldRedactPhone() {
        // Given
        String text = "Call me at +1-555-123-4567 or (555) 987-6543";

        // When
        String redacted = redactionService.redact(text);

        // Then
        assertThat(redacted).contains("[REDACTED_PHONE]");
    }

    @Test
    @Order(3)
    @DisplayName("Should redact credit card numbers")
    void shouldRedactCreditCard() {
        // Given
        String text = "My card number is 4532-1234-5678-9012";

        // When
        String redacted = redactionService.redact(text);

        // Then
        assertThat(redacted).contains("[REDACTED_CC]");
        assertThat(redacted).doesNotContain("4532-1234-5678-9012");
    }

    @Test
    @Order(4)
    @DisplayName("Should redact SSN")
    void shouldRedactSSN() {
        // Given
        String text = "SSN: 123-45-6789";

        // When
        String redacted = redactionService.redact(text);

        // Then
        assertThat(redacted).contains("[REDACTED_SSN]");
        assertThat(redacted).doesNotContain("123-45-6789");
    }

    @Test
    @Order(5)
    @DisplayName("Should redact IP addresses")
    void shouldRedactIPAddress() {
        // Given
        String text = "Server IP: 192.168.1.100";

        // When
        String redacted = redactionService.redact(text);

        // Then
        assertThat(redacted).contains("[REDACTED_IP]");
        assertThat(redacted).doesNotContain("192.168.1.100");
    }

    @Test
    @Order(6)
    @DisplayName("Should redact API keys")
    void shouldRedactAPIKey() {
        // Given
        String text = "api_key = sk_test_abcdefghijklmnopqrstuvwxyz123456";

        // When
        String redacted = redactionService.redact(text);

        // Then
        assertThat(redacted).contains("[REDACTED_API_KEY]");
    }

    @Test
    @Order(7)
    @DisplayName("Should redact AWS access keys")
    void shouldRedactAWSKey() {
        // Given
        String text = "AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE";

        // When
        String redacted = redactionService.redact(text);

        // Then
        assertThat(redacted).contains("[REDACTED_AWS_KEY]");
        assertThat(redacted).doesNotContain("AKIAIOSFODNN7EXAMPLE");
    }

    @Test
    @Order(8)
    @DisplayName("Should detect PII without redacting")
    void shouldDetectPII() {
        // Given
        String text = "Email: test@example.com, Phone: 555-123-4567";

        // When
        var detections = redactionService.detectPII(text);

        // Then
        assertThat(detections).isNotEmpty();
        assertThat(detections).containsKey("email");
        assertThat(detections).containsKey("phone");
    }

    @Test
    @Order(9)
    @DisplayName("Should handle null and empty input")
    void shouldHandleNullOrEmptyInput() {
        // When
        String nullResult = redactionService.redact(null);
        String emptyResult = redactionService.redact("");

        // Then
        assertThat(nullResult).isNull();
        assertThat(emptyResult).isEmpty();
    }

    @Test
    @Order(10)
    @DisplayName("Should track redaction statistics")
    void shouldTrackStatistics() {
        // Given
        redactionService.clearStats();
        String text = "Email: test@example.com";

        // When
        redactionService.redact(text);
        var stats = redactionService.getRedactionStats();

        // Then
        assertThat(stats).isNotEmpty();
        assertThat(stats).containsKey("email");
        assertThat(stats.get("email")).isGreaterThan(0);
    }

    @Test
    @Order(11)
    @DisplayName("Should add custom patterns")
    void shouldAddCustomPattern() {
        // Given
        redactionService.addPattern("custom", "\\bSECRET\\d+\\b", "[REDACTED_CUSTOM]", true);
        String text = "My SECRET123 should be hidden";

        // When
        String redacted = redactionService.redact(text);

        // Then
        assertThat(redacted).contains("[REDACTED_CUSTOM]");
        assertThat(redacted).doesNotContain("SECRET123");
    }

    @Test
    @Order(12)
    @DisplayName("Should enable/disable patterns")
    void shouldEnableDisablePattern() {
        // Given
        redactionService.setPatternEnabled("email", false);
        String text = "Contact: test@example.com";

        // When
        String redacted = redactionService.redact(text);

        // Then
        assertThat(redacted).doesNotContain("[REDACTED_EMAIL]");
        assertThat(redacted).contains("test@example.com");

        // Re-enable
        redactionService.setPatternEnabled("email", true);
        String redacted2 = redactionService.redact(text);
        assertThat(redacted2).contains("[REDACTED_EMAIL]");
    }

    @Test
    @Order(13)
    @DisplayName("Should return pattern info")
    void shouldGetPatterns() {
        // When
        var patterns = redactionService.getPatterns();

        // Then
        assertThat(patterns).isNotEmpty();
        assertThat(patterns).containsKey("email");
        assertThat(patterns).containsKey("phone");
        assertThat(patterns.get("email").enabled).isTrue();
    }

    @Test
    @Order(14)
    @DisplayName("Should handle multiple PII types in one text")
    void shouldHandleMultiplePIITypes() {
        // Given
        String text = "User john@example.com with SSN 123-45-6789 and card 4532-1234-5678-9012";

        // When
        String redacted = redactionService.redact(text);

        // Then
        assertThat(redacted).contains("[REDACTED_EMAIL]");
        assertThat(redacted).contains("[REDACTED_SSN]");
        assertThat(redacted).contains("[REDACTED_CC]");
        assertThat(redacted).doesNotContain("john@example.com");
        assertThat(redacted).doesNotContain("123-45-6789");
        assertThat(redacted).doesNotContain("4532-1234-5678-9012");
    }
}
