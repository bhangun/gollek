package tech.kayys.gollek.safetensor.engine.tooling;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

import java.time.Instant;
import java.time.ZoneOffset;
import java.time.YearMonth;

public class MultiTenantQuotaServiceTest {

    private MultiTenantQuotaService quotaService;

    @BeforeEach
    void setUp() {
        quotaService = new MultiTenantQuotaService();
        quotaService.enabled = true;
        quotaService.defaultMonthlyTokens = 1000;
        // Mocking mpConfig if needed, but for now we test default behavior
    }

    @Test
    void testPeriodEndCalculation() {
        String tenantId = "test-tenant";
        MultiTenantQuotaService.QuotaStats stats = quotaService.stats(tenantId);
        
        Instant start = stats.periodStart();
        Instant end = stats.periodEnd();
        
        assertNotNull(start);
        assertNotNull(end);
        assertTrue(end.isAfter(start));
        
        // Verify it's roughly one month later
        assertTrue(end.toEpochMilli() - start.toEpochMilli() > 27L * 24 * 3600 * 1000);
    }
}
