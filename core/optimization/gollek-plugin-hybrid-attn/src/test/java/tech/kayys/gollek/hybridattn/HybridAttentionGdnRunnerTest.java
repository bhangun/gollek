package tech.kayys.gollek.hybridattn;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class HybridAttentionGdnRunnerTest {

    @Test
    void buildLayerScheduleRepeatsH2Pattern() {
        boolean[] sched = HybridAttentionGdnRunner.buildLayerSchedule("H2", 12);
        assertEquals(12, sched.length);
        assertTrue(sched[0]);
        assertTrue(sched[1]);
        assertFalse(sched[2]);
        assertTrue(sched[3]);
        assertTrue(sched[4]);
        assertFalse(sched[5]);
        assertTrue(sched[6]);
        assertTrue(sched[7]);
        assertFalse(sched[8]);
        assertTrue(sched[9]);
        assertTrue(sched[10]);
        assertFalse(sched[11]);
    }

    @Test
    void buildLayerScheduleDefaultsToH2() {
        boolean[] sched = HybridAttentionGdnRunner.buildLayerSchedule("unknown", 6);
        assertEquals(6, sched.length);
        assertTrue(sched[0]);
        assertTrue(sched[1]);
        assertFalse(sched[2]);
        assertTrue(sched[3]);
        assertTrue(sched[4]);
        assertFalse(sched[5]);
    }
}
