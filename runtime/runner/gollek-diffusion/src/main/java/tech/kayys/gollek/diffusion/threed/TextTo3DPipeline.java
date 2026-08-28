package tech.kayys.gollek.diffusion.threed;

import tech.kayys.alkhawarizm.threed.export.GltfExporter;
import tech.kayys.alkhawarizm.threed.export.ObjExporter;
import tech.kayys.alkhawarizm.threed.export.PlyExporter;
import tech.kayys.alkhawarizm.threed.geometry.Mesh3D;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Objects;

/**
 * End-to-end Text-to-3D pipeline powered by Block3D diffusion framework.
 */
public final class TextTo3DPipeline {

    private final Block3DGenerator generator;
    private final ShapeDecoder3D decoder;

    public TextTo3DPipeline(Block3DGenerator generator, ShapeDecoder3D decoder) {
        this.generator = generator != null ? generator : Block3DGenerator.create();
        this.decoder = decoder != null ? decoder : ShapeDecoder3D.createDefault();
    }

    public static TextTo3DPipeline createDefault() {
        return new TextTo3DPipeline(Block3DGenerator.create(), ShapeDecoder3D.createDefault());
    }

    /**
     * Generate a 3D polygonal mesh from text prompt.
     *
     * @param prompt Description of 3D object to generate.
     * @return Watertight Mesh3D.
     */
    public Mesh3D generateMesh(String prompt) {
        Objects.requireNonNull(prompt, "prompt must not be null");
        int[] shapeCodes = generator.generate(prompt);
        String name = prompt.replaceAll("[^a-zA-Z0-9_]+", "_").toLowerCase();
        if (name.length() > 30) name = name.substring(0, 30);
        return decoder.decode(shapeCodes, name);
    }

    /**
     * Generate 3D object and export to Wavefront OBJ format.
     */
    public void generateAndExportObj(String prompt, Path outputPath) throws IOException {
        Mesh3D mesh = generateMesh(prompt);
        ObjExporter.exportToFile(mesh, outputPath);
    }

    /**
     * Generate 3D object and export to glTF / GLB format.
     */
    public void generateAndExportGltf(String prompt, Path outputPath) throws IOException {
        Mesh3D mesh = generateMesh(prompt);
        GltfExporter.exportToFile(mesh, outputPath);
    }

    /**
     * Generate 3D object and export to Stanford PLY format.
     */
    public void generateAndExportPly(String prompt, Path outputPath) throws IOException {
        Mesh3D mesh = generateMesh(prompt);
        PlyExporter.exportToFile(mesh, outputPath);
    }
}
