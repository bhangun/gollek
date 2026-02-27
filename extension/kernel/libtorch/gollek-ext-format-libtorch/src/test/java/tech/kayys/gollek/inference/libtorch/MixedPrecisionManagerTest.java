package tech.kayys.gollek.inference.libtorch;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for {@link MixedPrecisionManager}.
 */
class MixedPrecisionManagerTest {

    @Test
    void precisionFromStringFp16() {
        assertEquals(MixedPrecisionManager.Precision.FP16,
                MixedPrecisionManager.Precision.fromString("fp16"));
        assertEquals(MixedPrecisionManager.Precision.FP16,
                MixedPrecisionManager.Precision.fromString("float16"));
        assertEquals(MixedPrecisionManager.Precision.FP16,
                MixedPrecisionManager.Precision.fromString("half"));
        assertEquals(MixedPrecisionManager.Precision.FP16,
                MixedPrecisionManager.Precision.fromString("FP16"));
    }

    @Test
    void precisionFromStringBf16() {
        assertEquals(MixedPrecisionManager.Precision.BF16,
                MixedPrecisionManager.Precision.fromString("bf16"));
        assertEquals(MixedPrecisionManager.Precision.BF16,
                MixedPrecisionManager.Precision.fromString("bfloat16"));
    }

    @Test
    void precisionFromStringFp32Default() {
        assertEquals(MixedPrecisionManager.Precision.FP32,
                MixedPrecisionManager.Precision.fromString("fp32"));
        assertEquals(MixedPrecisionManager.Precision.FP32,
                MixedPrecisionManager.Precision.fromString("float32"));
        assertEquals(MixedPrecisionManager.Precision.FP32,
                MixedPrecisionManager.Precision.fromString("unknown"));
    }

    @Test
    void precisionFromStringNullReturnsFp32() {
        assertEquals(MixedPrecisionManager.Precision.FP32,
                MixedPrecisionManager.Precision.fromString(null));
    }

    @Test
    void precisionTorchDtype() {
        assertEquals("float32", MixedPrecisionManager.Precision.FP32.getTorchDtype());
        assertEquals("float16", MixedPrecisionManager.Precision.FP16.getTorchDtype());
        assertEquals("bfloat16", MixedPrecisionManager.Precision.BF16.getTorchDtype());
    }

    @Test
    void defaultPrecisionIsFp32() {
        MixedPrecisionManager manager = new MixedPrecisionManager();
        assertEquals(MixedPrecisionManager.Precision.FP32, manager.getActivePrecision());
    }

    @Test
    void notMixedPrecisionByDefault() {
        MixedPrecisionManager manager = new MixedPrecisionManager();
        assertFalse(manager.isMixedPrecisionActive());
    }
}
