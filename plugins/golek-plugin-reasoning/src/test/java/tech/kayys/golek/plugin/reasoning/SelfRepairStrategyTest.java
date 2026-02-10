package tech.kayys.golek.plugin.reasoning;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for {@link SelfRepairStrategy}.
 */
class SelfRepairStrategyTest {

    private SelfRepairStrategy strategy;

    @BeforeEach
    void setUp() {
        strategy = new SelfRepairStrategy();
    }

    @Test
    void repair_closesUnclosedJson() {
        String malformed = "{\"tool_call\": {\"name\": \"test\"";

        Optional<String> repaired = strategy.repair(malformed);

        assertTrue(repaired.isPresent());
        String result = repaired.get();
        assertTrue(result.endsWith("}}"));
    }

    @Test
    void repair_returnsEmptyForNull() {
        assertTrue(strategy.repair(null).isEmpty());
        assertTrue(strategy.repair("").isEmpty());
    }

    @Test
    void repair_closesUnclosedString() {
        String malformed = "{\"name\": \"test";

        Optional<String> repaired = strategy.repair(malformed);

        assertTrue(repaired.isPresent());
        assertTrue(repaired.get().contains("\""));
    }

    @Test
    void repair_returnsEmptyForValidText() {
        // Already valid, nothing to fix
        Optional<String> result = strategy.repair("Just normal text");
        assertTrue(result.isEmpty());
    }
}
