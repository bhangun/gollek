package tech.kayys.gollek.onnx.runner;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import javax.imageio.ImageIO;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Objects;

public final class PaddleOcrVlOnnxPlanner {
    private static final ObjectMapper MAPPER = new ObjectMapper();

    private PaddleOcrVlOnnxPlanner() {
    }

    public static Plan plan(Path modelDir, List<Path> images, String requestedVariant) {
        Objects.requireNonNull(modelDir, "modelDir");
        Path normalizedModelDir = modelDir.toAbsolutePath().normalize();
        ProcessorConfig processor = ProcessorConfig.load(normalizedModelDir);
        GraphSelection graphs = selectGraphs(normalizedModelDir, requestedVariant);
        List<ImagePlan> imagePlans = new ArrayList<>();
        if (images != null) {
            for (Path image : images) {
                if (image != null) {
                    imagePlans.add(planImage(image, processor));
                }
            }
        }
        return new Plan(normalizedModelDir, graphs, processor, List.copyOf(imagePlans));
    }

    public static ImagePlan planImage(Path imagePath, ProcessorConfig processor) {
        Objects.requireNonNull(imagePath, "imagePath");
        Objects.requireNonNull(processor, "processor");
        Path normalized = imagePath.toAbsolutePath().normalize();
        BufferedImage image = readImage(normalized);

        int originalWidth = image.getWidth();
        int originalHeight = image.getHeight();
        Resize resize = smartResize(
                originalHeight,
                originalWidth,
                processor.patchSize() * processor.mergeSize(),
                processor.minPixels(),
                processor.maxPixels());
        int gridT = 1;
        int gridH = resize.height() / processor.patchSize();
        int gridW = resize.width() / processor.patchSize();
        int patchCount = gridT * gridH * gridW;
        int promptImageTokens = Math.max(1, patchCount / Math.max(1, processor.mergeSize() * processor.mergeSize()));

        return new ImagePlan(
                normalized,
                originalWidth,
                originalHeight,
                resize.width(),
                resize.height(),
                gridT,
                gridH,
                gridW,
                patchCount,
                promptImageTokens,
                processor.patchSize(),
                processor.mergeSize(),
                (long) originalWidth * originalHeight,
                (long) resize.width() * resize.height());
    }

    public static ImageTensor preprocessImage(Path imagePath, ProcessorConfig processor) {
        if (processor.temporalPatchSize() != 1) {
            throw new IllegalArgumentException("temporal_patch_size=" + processor.temporalPatchSize()
                    + " is not supported by the Java PaddleOCR-VL image patchifier yet");
        }
        ImagePlan imagePlan = planImage(imagePath, processor);
        BufferedImage image = readImage(imagePlan.path());
        BufferedImage resized = resizeRgb(image, imagePlan.resizedWidth(), imagePlan.resizedHeight());
        int patchSize = processor.patchSize();
        int channels = 3;
        float[] values = new float[Math.multiplyExact(
                imagePlan.patchCount(),
                channels * patchSize * patchSize)];
        double[] mean = channelValues(processor.imageMean(), 0.5);
        double[] std = channelValues(processor.imageStd(), 0.5);

        float min = Float.POSITIVE_INFINITY;
        float max = Float.NEGATIVE_INFINITY;
        double sum = 0.0;
        int out = 0;
        for (int gridH = 0; gridH < imagePlan.gridH(); gridH++) {
            for (int gridW = 0; gridW < imagePlan.gridW(); gridW++) {
                for (int channel = 0; channel < channels; channel++) {
                    for (int patchY = 0; patchY < patchSize; patchY++) {
                        int y = gridH * patchSize + patchY;
                        for (int patchX = 0; patchX < patchSize; patchX++) {
                            int x = gridW * patchSize + patchX;
                            int rgb = resized.getRGB(x, y);
                            int raw = switch (channel) {
                                case 0 -> (rgb >>> 16) & 0xff;
                                case 1 -> (rgb >>> 8) & 0xff;
                                default -> rgb & 0xff;
                            };
                            float normalized = (float) ((raw * processor.rescaleFactor() - mean[channel]) / std[channel]);
                            values[out++] = normalized;
                            min = Math.min(min, normalized);
                            max = Math.max(max, normalized);
                            sum += normalized;
                        }
                    }
                }
            }
        }
        double average = values.length == 0 ? 0.0 : sum / values.length;
        return new ImageTensor(
                imagePlan,
                values,
                new long[] { 1, imagePlan.patchCount(), channels, patchSize, patchSize },
                new long[] { 1, 3 },
                new long[] { imagePlan.gridT(), imagePlan.gridH(), imagePlan.gridW() },
                (long) values.length * Float.BYTES,
                checksum(values),
                min,
                max,
                average);
    }

    private static BufferedImage readImage(Path path) {
        if (!Files.isRegularFile(path)) {
            throw new IllegalArgumentException("input image not found: " + path);
        }
        try {
            BufferedImage image = ImageIO.read(path.toFile());
            if (image == null) {
                throw new IllegalArgumentException("unsupported image format: " + path);
            }
            return image;
        } catch (IOException e) {
            throw new IllegalArgumentException("failed to read image: " + path + " (" + e.getMessage() + ")", e);
        }
    }

    private static BufferedImage resizeRgb(BufferedImage image, int width, int height) {
        BufferedImage target = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics2D graphics = target.createGraphics();
        try {
            graphics.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BICUBIC);
            graphics.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_QUALITY);
            graphics.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
            graphics.drawImage(image, 0, 0, width, height, null);
        } finally {
            graphics.dispose();
        }
        return target;
    }

    private static double[] channelValues(List<Double> values, double fallback) {
        double[] channels = { fallback, fallback, fallback };
        if (values == null) {
            return channels;
        }
        for (int i = 0; i < Math.min(channels.length, values.size()); i++) {
            Double value = values.get(i);
            channels[i] = value == null ? fallback : value;
        }
        return channels;
    }

    private static String checksum(float[] values) {
        long hash = 0xcbf29ce484222325L;
        for (float value : values) {
            int bits = Float.floatToIntBits(value);
            for (int shift = 0; shift < Integer.SIZE; shift += Byte.SIZE) {
                hash ^= (bits >>> shift) & 0xffL;
                hash *= 0x100000001b3L;
            }
        }
        return String.format(Locale.ROOT, "%016x", hash);
    }

    private static Resize smartResize(int height, int width, int factor, int minPixels, int maxPixels) {
        if (height <= 0 || width <= 0) {
            throw new IllegalArgumentException("image dimensions must be positive");
        }
        int adjustedHeight = height;
        int adjustedWidth = width;
        if (adjustedHeight < factor) {
            adjustedWidth = Math.max(1, Math.round((adjustedWidth * factor) / (float) adjustedHeight));
            adjustedHeight = factor;
        }
        if (adjustedWidth < factor) {
            adjustedHeight = Math.max(1, Math.round((adjustedHeight * factor) / (float) adjustedWidth));
            adjustedWidth = factor;
        }

        double aspect = Math.max(adjustedHeight, adjustedWidth) / (double) Math.min(adjustedHeight, adjustedWidth);
        if (aspect > 200.0) {
            throw new IllegalArgumentException("absolute aspect ratio must be smaller than 200, got " + aspect);
        }

        int hBar = Math.round(adjustedHeight / (float) factor) * factor;
        int wBar = Math.round(adjustedWidth / (float) factor) * factor;
        long resizedPixels = (long) hBar * wBar;
        if (resizedPixels > maxPixels) {
            double beta = Math.sqrt((adjustedHeight * (double) adjustedWidth) / maxPixels);
            hBar = (int) Math.floor(adjustedHeight / beta / factor) * factor;
            wBar = (int) Math.floor(adjustedWidth / beta / factor) * factor;
        } else if (resizedPixels < minPixels) {
            double beta = Math.sqrt(minPixels / (adjustedHeight * (double) adjustedWidth));
            hBar = (int) Math.ceil(adjustedHeight * beta / factor) * factor;
            wBar = (int) Math.ceil(adjustedWidth * beta / factor) * factor;
        }
        hBar = Math.max(factor, hBar);
        wBar = Math.max(factor, wBar);
        return new Resize(hBar, wBar);
    }

    private static GraphSelection selectGraphs(Path modelDir, String requestedVariant) {
        Path onnxDir = modelDir.resolve("onnx");
        String variant = normalizeVariant(requestedVariant);
        List<String> warnings = new ArrayList<>();
        Path vision = selectRoleGraph(onnxDir, "vision_encoder", variant, warnings);
        Path decoder = selectRoleGraph(onnxDir, "decoder", variant, warnings);
        Path embedding = onnxDir.resolve("embedding.onnx");
        if (!Files.isRegularFile(embedding)) {
            warnings.add("embedding.onnx was not found");
        }
        return new GraphSelection(
                variant,
                vision,
                embedding,
                decoder,
                List.copyOf(warnings));
    }

    private static Path selectRoleGraph(Path onnxDir, String role, String variant, List<String> warnings) {
        List<String> candidates = candidateNames(role, variant);
        for (String name : candidates) {
            Path path = onnxDir.resolve(name);
            if (Files.isRegularFile(path)) {
                return path.toAbsolutePath().normalize();
            }
        }
        warnings.add(role + " graph not found for variant " + variant + " (tried " + String.join(", ", candidates) + ")");
        return onnxDir.resolve(candidates.get(candidates.size() - 1)).toAbsolutePath().normalize();
    }

    private static List<String> candidateNames(String role, String variant) {
        String full = role + ".onnx";
        String q8 = role + "_q8.onnx";
        String q4 = role + "_q4.onnx";
        if ("decoder".equals(role) && "uint8".equals(variant)) {
            return List.of("decoder_quint8.onnx", q8, q4, full);
        }
        return switch (variant) {
            case "full" -> List.of(full, q8, q4);
            case "q8" -> List.of(q8, full, q4);
            case "q4", "auto" -> List.of(q4, q8, full);
            default -> List.of(q4, q8, full);
        };
    }

    public static String normalizeVariant(String requestedVariant) {
        if (requestedVariant == null || requestedVariant.isBlank()) {
            return "auto";
        }
        String value = requestedVariant.trim().toLowerCase(Locale.ROOT);
        return switch (value) {
            case "fp32", "float32", "f32" -> "full";
            case "int8", "8bit" -> "q8";
            case "int4", "4bit", "small", "fast" -> "q4";
            case "quint8", "u8" -> "uint8";
            case "auto", "full", "q8", "q4", "uint8" -> value;
            default -> "auto";
        };
    }

    private record Resize(int height, int width) {
    }

    public record Plan(
            Path modelDir,
            GraphSelection graphs,
            ProcessorConfig processor,
            List<ImagePlan> images) {
        public int totalPromptImageTokens() {
            int total = 0;
            for (ImagePlan image : images) {
                total += image.promptImageTokens();
            }
            return total;
        }
    }

    public record GraphSelection(
            String variant,
            Path visionEncoder,
            Path embedding,
            Path decoder,
            List<String> warnings) {
    }

    public record ProcessorConfig(
            int patchSize,
            int mergeSize,
            int temporalPatchSize,
            int minPixels,
            int maxPixels,
            double rescaleFactor,
            List<Double> imageMean,
            List<Double> imageStd) {

        static ProcessorConfig load(Path modelDir) {
            Path configPath = modelDir.resolve("processor_config.json");
            JsonNode imageProcessor = null;
            if (Files.isRegularFile(configPath)) {
                try {
                    JsonNode root = MAPPER.readTree(configPath.toFile());
                    imageProcessor = root.path("image_processor");
                } catch (Exception ignored) {
                    imageProcessor = null;
                }
            }
            if (imageProcessor == null || imageProcessor.isMissingNode() || imageProcessor.isNull()) {
                imageProcessor = MAPPER.createObjectNode();
            }
            return new ProcessorConfig(
                    intValue(imageProcessor, "patch_size", 14),
                    intValue(imageProcessor, "merge_size", 2),
                    intValue(imageProcessor, "temporal_patch_size", 1),
                    intValue(imageProcessor, "min_pixels", 112_896),
                    intValue(imageProcessor, "max_pixels", 1_003_520),
                    doubleValue(imageProcessor, "rescale_factor", 1.0 / 255.0),
                    doubleList(imageProcessor.path("image_mean"), List.of(0.5, 0.5, 0.5)),
                    doubleList(imageProcessor.path("image_std"), List.of(0.5, 0.5, 0.5)));
        }

        private static int intValue(JsonNode node, String field, int fallback) {
            JsonNode value = node.path(field);
            return value.isNumber() ? value.asInt() : fallback;
        }

        private static double doubleValue(JsonNode node, String field, double fallback) {
            JsonNode value = node.path(field);
            return value.isNumber() ? value.asDouble() : fallback;
        }

        private static List<Double> doubleList(JsonNode node, List<Double> fallback) {
            if (node == null || !node.isArray()) {
                return fallback;
            }
            List<Double> values = new ArrayList<>();
            for (JsonNode item : node) {
                if (item.isNumber()) {
                    values.add(item.asDouble());
                }
            }
            return values.isEmpty() ? fallback : List.copyOf(values);
        }
    }

    public record ImagePlan(
            Path path,
            int originalWidth,
            int originalHeight,
            int resizedWidth,
            int resizedHeight,
            int gridT,
            int gridH,
            int gridW,
            int patchCount,
            int promptImageTokens,
            int patchSize,
            int mergeSize,
            long originalPixels,
            long resizedPixels) {
    }

    public record ImageTensor(
            ImagePlan image,
            float[] pixelValues,
            long[] pixelValuesShape,
            long[] imageGridThwShape,
            long[] imageGridThw,
            long byteSize,
            String checksum,
            float min,
            float max,
            double mean) {
    }
}
