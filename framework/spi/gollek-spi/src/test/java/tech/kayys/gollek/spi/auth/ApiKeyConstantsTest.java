package tech.kayys.gollek.spi.auth;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class ApiKeyConstantsTest {

    @Test
    void authorizationValueUsesBearerScheme() {
        assertEquals("Bearer test-key", ApiKeyConstants.authorizationValue("test-key"));
    }
}
