package tech.kayys.golek.cli.commands;

import io.quarkus.arc.Unremovable;
import jakarta.enterprise.context.Dependent;
import jakarta.json.Json;
import jakarta.json.JsonObject;
import jakarta.json.JsonReader;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;

import java.io.StringReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Map;

@Dependent
@Unremovable
@Command(name = "safetensors", mixinStandardHelpOptions = true, description = "Inspect safetensors metadata")
public class SafetensorsCommand implements Runnable {

    @Option(names = { "--path" }, description = "Path to .safetensors file")
    String path;

    @Option(names = { "--model" }, description = "Model ID under ~/.golek/models/torchscript (e.g. Qwen/Qwen2.5-0.5B-Instruct)")
    String modelId;

    @Option(names = { "--limit" }, description = "Maximum tensors to print", defaultValue = "30")
    int limit;

    @Override
    public void run() {
        try {
            Path file = resolveInputPath();
            if (file == null) {
                System.err.println("Error: provide either --path or --model.");
                return;
            }
            if (!Files.exists(file)) {
                System.err.println("Error: safetensors file not found: " + file);
                return;
            }

            Map<String, TensorMetadata> metadata = parse(file);

            long totalBytes = metadata.values().stream().mapToLong(TensorMetadata::length).sum();
            System.out.println("File: " + file);
            System.out.println("Tensors: " + metadata.size());
            System.out.printf("Payload: %.2f MB%n", totalBytes / (1024.0 * 1024.0));

            int toPrint = Math.max(0, Math.min(limit, metadata.size()));
            if (toPrint == 0) {
                return;
            }

            System.out.println("----");
            metadata.entrySet().stream()
                    .sorted(Comparator.comparing(Map.Entry::getKey))
                    .limit(toPrint)
                    .forEach(entry -> {
                        TensorMetadata info = entry.getValue();
                        System.out.printf("%s | dtype=%s | shape=%s | bytes=%d%n",
                                entry.getKey(),
                                info.dtype(),
                                java.util.Arrays.toString(info.shape()),
                                info.length());
                    });

            if (metadata.size() > toPrint) {
                System.out.printf("... (%d more tensors)%n", metadata.size() - toPrint);
            }
        } catch (Exception e) {
            System.err.println("Failed to inspect safetensors: " + e.getMessage());
        }
    }

    private Path resolveInputPath() {
        if (path != null && !path.isBlank()) {
            return Path.of(path).toAbsolutePath().normalize();
        }
        if (modelId == null || modelId.isBlank()) {
            return null;
        }

        Path base = Path.of(System.getProperty("user.home"), ".golek", "models", "torchscript");
        Path direct = base.resolve(modelId + ".safetensors");
        if (Files.exists(direct)) {
            return direct;
        }

        Path nested = base.resolve(modelId).resolve("model.safetensors");
        if (Files.exists(nested)) {
            return nested;
        }

        String normalized = modelId.replace("/", "_");
        Path normalizedPath = base.resolve(normalized + ".safetensors");
        if (Files.exists(normalizedPath)) {
            return normalizedPath;
        }

        // fallback: first safetensors under model dir
        Path modelDir = base.resolve(modelId);
        if (Files.isDirectory(modelDir)) {
            try (var stream = Files.walk(modelDir, 2)) {
                return stream
                        .filter(Files::isRegularFile)
                        .filter(p -> p.getFileName().toString().endsWith(".safetensors"))
                        .findFirst()
                        .orElse(direct);
            } catch (Exception ignored) {
                return direct;
            }
        }
        return direct;
    }

    private Map<String, TensorMetadata> parse(Path path) throws Exception {
        byte[] all = Files.readAllBytes(path);
        if (all.length < 8) {
            throw new IllegalArgumentException("File too small to be safetensors");
        }

        ByteBuffer bb = ByteBuffer.wrap(all, 0, 8).order(ByteOrder.LITTLE_ENDIAN);
        long headerLength = bb.getLong();
        if (headerLength <= 0 || headerLength > all.length - 8) {
            throw new IllegalArgumentException("Invalid safetensors header length: " + headerLength);
        }

        String headerJson = new String(all, 8, (int) headerLength, StandardCharsets.UTF_8);
        long baseOffset = 8 + headerLength;

        Map<String, TensorMetadata> out = new HashMap<>();
        try (JsonReader reader = Json.createReader(new StringReader(headerJson))) {
            JsonObject root = reader.readObject();
            for (String key : root.keySet()) {
                if ("__metadata__".equals(key)) {
                    continue;
                }
                JsonObject entry = root.getJsonObject(key);
                String dtype = entry.getString("dtype");
                long[] shape = entry.getJsonArray("shape")
                        .stream()
                        .mapToLong(v -> ((jakarta.json.JsonNumber) v).longValue())
                        .toArray();
                var offsets = entry.getJsonArray("data_offsets");
                long start = ((jakarta.json.JsonNumber) offsets.get(0)).longValue();
                long end = ((jakarta.json.JsonNumber) offsets.get(1)).longValue();
                out.put(key, new TensorMetadata(dtype, shape, baseOffset + start, end - start));
            }
        }
        return out;
    }

    private record TensorMetadata(String dtype, long[] shape, long absoluteStart, long length) {
    }
}
