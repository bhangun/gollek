package tech.kayys.gollek.spi.image;

import org.junit.jupiter.api.Test;
import java.util.Map;
import static org.junit.jupiter.api.Assertions.*;

public class ImageGenRequestTest {

    @Test
    void testDefaultValues() {
        ImageGenRequest req = ImageGenRequest.builder()
                .prompt("A futuristic city in watercolor")
                .build();

        assertNotNull(req.requestId());
        assertEquals("A futuristic city in watercolor", req.prompt());
        assertEquals("", req.negativePrompt());
        assertEquals(512, req.width());
        assertEquals(512, req.height());
        assertEquals(20, req.steps());
        assertEquals(7.5f, req.guidanceScale());
        assertEquals("default", req.scheduler());
        assertEquals("png", req.outputFormat());
        assertFalse(req.isImageToImage());
        assertFalse(req.isInpainting());
    }

    @Test
    void testImageToImageAndInpainting() {
        byte[] dummyInit = new byte[]{1, 2, 3};
        byte[] dummyMask = new byte[]{4, 5, 6};

        ImageGenRequest img2img = ImageGenRequest.builder()
                .prompt("Turn into oil painting")
                .initImage(dummyInit)
                .strength(0.6f)
                .build();

        assertTrue(img2img.isImageToImage());
        assertFalse(img2img.isInpainting());
        assertEquals(0.6f, img2img.strength());

        ImageGenRequest inpaint = ImageGenRequest.builder()
                .prompt("Add sunglasses")
                .initImage(dummyInit)
                .maskImage(dummyMask)
                .build();

        assertFalse(inpaint.isImageToImage());
        assertTrue(inpaint.isInpainting());
    }

    @Test
    void testCustomParameters() {
        ImageGenRequest req = ImageGenRequest.builder()
                .requestId("req-123")
                .prompt("A high-tech robot")
                .negativePrompt("blurry, low quality")
                .dimensions(1024, 768)
                .steps(4)
                .guidanceScale(3.5f)
                .seed(42L)
                .scheduler("flow_euler")
                .outputFormat("webp")
                .extras(Map.of("fluxVariant", "klein-9B"))
                .build();

        assertEquals("req-123", req.requestId());
        assertEquals("A high-tech robot", req.prompt());
        assertEquals("blurry, low quality", req.negativePrompt());
        assertEquals(1024, req.width());
        assertEquals(768, req.height());
        assertEquals(4, req.steps());
        assertEquals(3.5f, req.guidanceScale());
        assertEquals(42L, req.seed());
        assertEquals("flow_euler", req.scheduler());
        assertEquals("webp", req.outputFormat());
        assertEquals("klein-9B", req.extras().get("fluxVariant"));
    }

    @Test
    void testGeneratedImageCreation() {
        byte[] dummy = new byte[]{1, 2, 3, 4};
        GeneratedImage img = GeneratedImage.ofPng("req-123", dummy, 1024, 1024, "flux-schnell", 1200L);

        assertEquals("req-123", img.requestId());
        assertArrayEquals(dummy, img.data());
        assertEquals("image/png", img.mimeType());
        assertEquals(1024, img.width());
        assertEquals(1024, img.height());
        assertEquals("flux-schnell", img.modelId());
        assertEquals(1200L, img.generationTimeMs());
    }
}
