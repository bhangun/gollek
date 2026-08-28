package tech.kayys.gollek.diffusion.threed;

import org.junit.jupiter.api.Test;
import tech.kayys.alkhawarizm.threed.export.ObjExporter;
import tech.kayys.alkhawarizm.threed.geometry.Mesh3D;

import static org.junit.jupiter.api.Assertions.*;

public class Block3DTest {

    @Test
    void testBlock3DGeneratorExecution() {
        Block3DConfig config = Block3DConfig.fastPreview();
        Block3DGenerator generator = new Block3DGenerator(config);

        int[] codes = generator.generate("a futuristic cyberpunk sports car");
        assertNotNull(codes);
        assertEquals(config.sequenceLength(), codes.length);

        // Verify no mask tokens remain in finalized sequence
        for (int code : codes) {
            assertNotEquals(config.maskTokenId(), code);
            assertTrue(code >= 0 && code < config.codebookSize());
        }
    }

    @Test
    void testTextTo3DEndToEndPipeline() {
        TextTo3DPipeline pipeline = TextTo3DPipeline.createDefault();
        Mesh3D mesh = pipeline.generateMesh("a medieval fantasy castle");

        assertNotNull(mesh);
        assertTrue(mesh.vertexCount() > 0);
        assertTrue(mesh.faceCount() > 0);

        String obj = ObjExporter.exportToString(mesh);
        assertTrue(obj.contains("v "));
        assertTrue(obj.contains("f "));
    }
}
