package tech.kayys.gollek.cli.commands;
import tech.kayys.gollek.sdk.route.*;
import tech.kayys.gollek.safetensor.engine.route.*;

import io.quarkus.arc.Unremovable;
import jakarta.enterprise.context.Dependent;
import jakarta.inject.Inject;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import tech.kayys.gollek.sdk.core.GollekSdk;
import tech.kayys.alkhawarizm.spi.model.MultimodalRequest;
import tech.kayys.alkhawarizm.spi.model.MultimodalResponse;
import tech.kayys.alkhawarizm.spi.model.MultimodalContent;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;

/**
 * Multimodal inference CLI command.
 * Supports image captioning, visual QA, image classification, and image embedding.
 *
 * Usage:
 *   gollek multimodal caption --image photo.jpg --model blip2
 *   gollek multimodal vqa --image photo.jpg --question "What is this?" --model vilt
 *   gollek multimodal classify --image photo.jpg --model vit
 *   gollek multimodal embed --image photo.jpg --model clip
 */
@Dependent
@Unremovable
@Command(name = "multimodal", description = "Run multimodal inference (image captioning, VQA, classification, embedding)",
        subcommands = {
            MultimodalCommand.CaptionCommand.class,
            MultimodalCommand.VqaCommand.class,
            MultimodalCommand.ClassifyCommand.class,
            MultimodalCommand.EmbedImageCommand.class,
        })
public class MultimodalCommand implements Runnable {

    private static final String RESET = "\u001B[0m";
    private static final String GREEN = "\u001B[32m";
    private static final String CYAN = "\u001B[36m";
    private static final String YELLOW = "\u001B[33m";
    private static final String DIM = "\u001B[2m";
    private static final String BOLD = "\u001B[1m";

    @Override
    public void run() {
        System.out.println(BOLD + "Gollek Multimodal Inference" + RESET);
        System.out.println();
        System.out.println("Available tasks:");
        System.out.println("  " + CYAN + "caption" + RESET + "    Generate a text caption from an image");
        System.out.println("  " + CYAN + "vqa" + RESET + "        Answer a question about an image (Visual QA)");
        System.out.println("  " + CYAN + "classify" + RESET + "   Classify an image");
        System.out.println("  " + CYAN + "embed" + RESET + "      Generate a vector embedding for an image");
        System.out.println();
        System.out.println("Example: " + DIM + "gollek multimodal caption --image photo.jpg --model blip2" + RESET);
        System.out.println();
        System.out.println("Run 'gollek multimodal <task> --help' for task-specific options.");
    }

    // =========================================================================
    // Shared helper methods
    // =========================================================================

    static byte[] loadImage(String imagePath) throws IOException {
        Path path = Path.of(imagePath);
        if (!Files.exists(path)) {
            throw new IOException("Image file not found: " + imagePath);
        }
        if (!Files.isReadable(path)) {
            throw new IOException("Image file not readable: " + imagePath);
        }
        long size = Files.size(path);
        if (size > 50 * 1024 * 1024) { // 50MB limit
            throw new IOException("Image file too large (max 50MB): " + size + " bytes");
        }
        return Files.readAllBytes(path);
    }

    static String detectMimeType(String imagePath) {
        String lower = imagePath.toLowerCase();
        if (lower.endsWith(".png")) return "image/png";
        if (lower.endsWith(".gif")) return "image/gif";
        if (lower.endsWith(".bmp")) return "image/bmp";
        if (lower.endsWith(".webp")) return "image/webp";
        if (lower.endsWith(".tiff") || lower.endsWith(".tif")) return "image/tiff";
        return "image/jpeg"; // default
    }

    static void printHeader(String task, String model, String imagePath) {
        System.out.println(BOLD + YELLOW + "Gollek Multimodal - " + task + RESET);
        System.out.printf(BOLD + "Model: " + RESET + CYAN + "%s" + RESET + "%n", model);
        System.out.printf(BOLD + "Image: " + RESET + "%s%n", imagePath);
        System.out.println(DIM + "-".repeat(50) + RESET);
    }

    static void printDuration(long startTime) {
        long duration = System.currentTimeMillis() - startTime;
        System.out.printf("%n" + DIM + "[Duration: %.2fs]" + RESET + "%n", duration / 1000.0);
    }

    static Map<String, Object> taskParams(String taskType) {
        Map<String, Object> params = new HashMap<>();
        params.put("task_type", taskType);
        return params;
    }

    static Map<String, Object> taskParams(String taskType, String key, Object value) {
        Map<String, Object> params = taskParams(taskType);
        params.put(key, value);
        return params;
    }

    // =========================================================================
    // Subcommand: caption
    // =========================================================================

    @Dependent
    @Unremovable
    @Command(name = "caption", description = "Generate a text caption from an image")
    public static class CaptionCommand implements Runnable {

        @Inject
        GollekSdk sdk;

        @Option(names = { "--image", "-i" }, description = "Path to the image file", required = true)
        String imagePath;

        @Option(names = { "--model", "-m" }, description = "Model ID (e.g. blip2, vit-gpt2)", required = true)
        String modelId;

        @Option(names = { "--provider" }, description = "Provider: onnx, safetensor, etc.")
        String providerId;

        @Override
        public void run() {
            long start = System.currentTimeMillis();
            try {
                printHeader("Image Captioning", modelId, imagePath);

                byte[] imageBytes = loadImage(imagePath);
                String mime = detectMimeType(imagePath);

                if (providerId != null && !providerId.isBlank()) {
                    
                }

                MultimodalRequest request = MultimodalRequest.builder()
                        .requestId(java.util.UUID.randomUUID().toString())
                        .model(modelId)
                        .inputs(MultimodalContent.ofBase64Image(imageBytes, mime))
                        .parameters(taskParams("IMAGE_CAPTIONING"))
                        .build();

                MultimodalResponse response = sdk.processMultimodal(request);
                String caption = response.getOutputs()[0].getText();

                System.out.println();
                System.out.println(GREEN + "Caption: " + RESET + caption);
                printDuration(start);

            } catch (Exception e) {
                System.err.println(YELLOW + "Captioning failed: " + RESET + e.getMessage());
            }
        }
    }

    // =========================================================================
    // Subcommand: vqa
    // =========================================================================

    @Dependent
    @Unremovable
    @Command(name = "vqa", description = "Answer a question about an image (Visual Question Answering)")
    public static class VqaCommand implements Runnable {

        @Inject
        GollekSdk sdk;

        @Option(names = { "--image", "-i" }, description = "Path to the image file", required = true)
        String imagePath;

        @Option(names = { "--question", "-q" }, description = "Question about the image", required = true)
        String question;

        @Option(names = { "--model", "-m" }, description = "Model ID (e.g. vilt, blip-vqa)", required = true)
        String modelId;

        @Option(names = { "--provider" }, description = "Provider: onnx, safetensor, etc.")
        String providerId;

        @Override
        public void run() {
            long start = System.currentTimeMillis();
            try {
                printHeader("Visual QA", modelId, imagePath);
                System.out.printf(BOLD + "Question: " + RESET + "%s%n", question);
                System.out.println(DIM + "-".repeat(50) + RESET);

                byte[] imageBytes = loadImage(imagePath);
                String mime = detectMimeType(imagePath);

                if (providerId != null && !providerId.isBlank()) {
                    
                }

                MultimodalRequest request = MultimodalRequest.builder()
                        .requestId(java.util.UUID.randomUUID().toString())
                        .model(modelId)
                        .inputs(MultimodalContent.ofBase64Image(imageBytes, mime))
                        .parameters(taskParams("VISUAL_QA", "question", question))
                        .build();

                MultimodalResponse response = sdk.processMultimodal(request);
                String answer = response.getOutputs()[0].getText();

                System.out.println();
                System.out.println(GREEN + "Answer: " + RESET + answer);
                printDuration(start);

            } catch (Exception e) {
                System.err.println(YELLOW + "VQA failed: " + RESET + e.getMessage());
            }
        }
    }

    // =========================================================================
    // Subcommand: classify
    // =========================================================================

    @Dependent
    @Unremovable
    @Command(name = "classify", description = "Classify an image into categories")
    public static class ClassifyCommand implements Runnable {

        @Inject
        GollekSdk sdk;

        @Option(names = { "--image", "-i" }, description = "Path to the image file", required = true)
        String imagePath;

        @Option(names = { "--model", "-m" }, description = "Model ID (e.g. vit-base, resnet)", required = true)
        String modelId;

        @Option(names = { "--top-k" }, description = "Number of top classifications to show", defaultValue = "5")
        int topK;

        @Option(names = { "--provider" }, description = "Provider: onnx, safetensor, etc.")
        String providerId;

        @Override
        public void run() {
            long start = System.currentTimeMillis();
            try {
                printHeader("Image Classification", modelId, imagePath);

                byte[] imageBytes = loadImage(imagePath);
                String mime = detectMimeType(imagePath);

                if (providerId != null && !providerId.isBlank()) {
                    
                }

                MultimodalRequest request = MultimodalRequest.builder()
                        .requestId(java.util.UUID.randomUUID().toString())
                        .model(modelId)
                        .inputs(MultimodalContent.ofBase64Image(imageBytes, mime))
                        .parameters(taskParams("IMAGE_CLASSIFICATION", "top_k", topK))
                        .build();

                MultimodalResponse response = sdk.processMultimodal(request);
                String result = response.getOutputs()[0].getText();

                System.out.println();
                System.out.println(GREEN + "Classification:" + RESET);
                for (String line : result.split("\n")) {
                    System.out.println("  " + line.trim());
                }
                printDuration(start);

            } catch (Exception e) {
                System.err.println(YELLOW + "Classification failed: " + RESET + e.getMessage());
            }
        }
    }

    // =========================================================================
    // Subcommand: embed (image)
    // =========================================================================

    @Dependent
    @Unremovable
    @Command(name = "embed", description = "Generate a vector embedding for an image")
    public static class EmbedImageCommand implements Runnable {

        @Inject
        GollekSdk sdk;

        @Option(names = { "--image", "-i" }, description = "Path to the image file", required = true)
        String imagePath;

        @Option(names = { "--model", "-m" }, description = "Model ID (e.g. clip, siglip)", required = true)
        String modelId;

        @Option(names = { "--provider" }, description = "Provider: onnx, safetensor, etc.")
        String providerId;

        @Option(names = { "--preview" }, description = "Show first N dimensions of embedding", defaultValue = "10")
        int previewDims;

        @Override
        public void run() {
            long start = System.currentTimeMillis();
            try {
                printHeader("Image Embedding", modelId, imagePath);

                byte[] imageBytes = loadImage(imagePath);
                String mime = detectMimeType(imagePath);

                if (providerId != null && !providerId.isBlank()) {
                    
                }

                MultimodalRequest request = MultimodalRequest.builder()
                        .requestId(java.util.UUID.randomUUID().toString())
                        .model(modelId)
                        .inputs(MultimodalContent.ofBase64Image(imageBytes, mime))
                        .parameters(taskParams("IMAGE_EMBEDDING"))
                        .build();

                MultimodalResponse response = sdk.processMultimodal(request);
                float[] embedding = response.getOutputs()[0].getEmbedding();

                System.out.println();
                System.out.printf(GREEN + "Embedding Dimension: " + RESET + "%d%n", embedding.length);

                // Preview first N dimensions
                int dims = Math.min(previewDims, embedding.length);
                StringBuilder preview = new StringBuilder("[");
                for (int i = 0; i < dims; i++) {
                    if (i > 0) preview.append(", ");
                    preview.append(String.format("%.6f", embedding[i]));
                }
                if (dims < embedding.length) {
                    preview.append(", ...");
                }
                preview.append("]");
                System.out.println(DIM + "Preview: " + preview + RESET);
                printDuration(start);

            } catch (Exception e) {
                System.err.println(YELLOW + "Image embedding failed: " + RESET + e.getMessage());
            }
        }
    }
}
