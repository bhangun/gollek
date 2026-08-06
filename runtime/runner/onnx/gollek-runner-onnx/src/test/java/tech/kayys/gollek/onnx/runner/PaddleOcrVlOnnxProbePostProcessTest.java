package tech.kayys.gollek.onnx.runner;

import org.junit.jupiter.api.Test;

import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class PaddleOcrVlOnnxProbePostProcessTest {

    @Test
    void groupsLocationTokensIntoNormalizedBoxes() {
        PaddleOcrVlOnnxProbe.OcrPostProcessResult result =
                PaddleOcrVlOnnxProbe.postProcessOcrText(
                        "<|LOC_262|><|LOC_262|><|LOC_236|><|LOC_191|>",
                        imagePlan());

        assertEquals(4, result.locations().size());
        assertEquals(1, result.boxes().size());
        PaddleOcrVlOnnxProbe.LocationBox box = result.boxes().getFirst();
        assertEquals(236, box.normalizedX1());
        assertEquals(191, box.normalizedY1());
        assertEquals(262, box.normalizedX2());
        assertEquals(262, box.normalizedY2());
        assertTrue(result.displayText().contains("box=[236,191,262,262]"));
        assertTrue(result.displayText().contains("pixels=[332,147,369,201]"));
    }

    @Test
    void associatesNearbyTextWithLocationRegion() {
        PaddleOcrVlOnnxProbe.OcrPostProcessResult result =
                PaddleOcrVlOnnxProbe.postProcessOcrText(
                        "Jakarta <|LOC_300|><|LOC_400|><|LOC_100|><|LOC_200|>",
                        imagePlan());

        assertEquals("Jakarta", result.textWithoutLocations());
        assertEquals(1, result.regions().size());
        PaddleOcrVlOnnxProbe.OcrTextRegion region = result.regions().getFirst();
        assertEquals("Jakarta", region.text());
        assertEquals(100, region.box().normalizedX1());
        assertEquals(200, region.box().normalizedY1());
        assertEquals(300, region.box().normalizedX2());
        assertEquals(400, region.box().normalizedY2());
        assertTrue(result.displayText().startsWith("OCR regions:"));
    }

    private static PaddleOcrVlOnnxPlanner.ImagePlan imagePlan() {
        return new PaddleOcrVlOnnxPlanner.ImagePlan(
                Path.of("/tmp/test.png"),
                1408,
                768,
                1344,
                728,
                1,
                52,
                96,
                1248,
                1248,
                14,
                2,
                1408L * 768L,
                1344L * 728L);
    }
}
