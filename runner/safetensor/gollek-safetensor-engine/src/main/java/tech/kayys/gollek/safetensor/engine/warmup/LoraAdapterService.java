package tech.kayys.gollek.safetensor.engine.warmup;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import tech.kayys.gollek.inference.libtorch.core.TorchTensor;
import tech.kayys.gollek.safetensor.loader.SafetensorLoaderFacade;
import tech.kayys.gollek.safetensor.loader.SafetensorShardLoader.SafetensorShardSession;
import tech.kayys.gollek.safetensor.loader.SafetensorTensor;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

import static tech.kayys.gollek.safetensor.engine.warmup.SafetensorJsonUtil.*;

/**
 * LoRA adapter loader and manager service.
 * Refactored from LoraAdapter to resolve build stabilization issues.
 */
@ApplicationScoped
public class LoraAdapterService {

    private static final Logger log = Logger.getLogger(LoraAdapterService.class);

    private static final String DEFAULT_ADAPTER_FILE = "adapter_model.safetensors";
    private static final String DEFAULT_CONFIG_FILE = "adapter_config.json";
    private static final String PEFT_ADAPTER_FILE = "adapter.safetensors";

    @Inject
    SafetensorLoaderFacade safetensorLoader;

    @Inject
    ObjectMapper objectMapper;

    private final Map<Path, LoadedAdapter> adapterCache = new ConcurrentHashMap<>();

    public LoadedAdapter load(Path adapterPath) {
        Objects.requireNonNull(adapterPath, "adapterPath must not be null");
        Path normalizedPath = adapterPath.toAbsolutePath().normalize();

        LoadedAdapter cached = adapterCache.get(normalizedPath);
        if (cached != null) {
            log.debugf("LoraAdapterService: returning cached adapter from %s", normalizedPath);
            return cached;
        }

        log.infof("LoraAdapterService: loading adapter from %s", normalizedPath);

        try {
            Path adapterFile = resolveAdapterFile(normalizedPath);
            if (!Files.isRegularFile(adapterFile)) {
                throw new IllegalArgumentException("Adapter file not found at " + adapterFile);
            }

            Map<String, TorchTensor> weights = loadAdapterWeights(adapterFile);
            if (weights.isEmpty()) {
                throw new IllegalArgumentException("No LoRA weights found in " + adapterFile);
            }

            AdapterConfig config = loadAdapterConfig(normalizedPath);
            validateRankConsistency(weights, config.rank());

            LoadedAdapter adapter = new LoadedAdapter(
                    normalizedPath,
                    weights,
                    config,
                    System.currentTimeMillis()
            );

            adapterCache.put(normalizedPath, adapter);
            log.infof("LoraAdapterService: loaded adapter — %d tensors, rank=%d, alpha=%.2f",
                    weights.size(), config.rank(), config.alpha());

            return adapter;

        } catch (IOException e) {
            throw new RuntimeException("Failed to load LoRA adapter from " + normalizedPath, e);
        }
    }

    public boolean isLoaded(Path adapterPath) {
        return adapterCache.containsKey(adapterPath.toAbsolutePath().normalize());
    }

    public Optional<LoadedAdapter> getCached(Path adapterPath) {
        return Optional.ofNullable(adapterCache.get(adapterPath.toAbsolutePath().normalize()));
    }

    public void unload(Path adapterPath) {
        Path normalizedPath = adapterPath.toAbsolutePath().normalize();
        LoadedAdapter adapter = adapterCache.remove(normalizedPath);
        if (adapter != null) {
            adapter.close();
            log.infof("LoraAdapterService: unloaded adapter from %s", normalizedPath);
        }
    }

    public void clearCache() {
        for (LoadedAdapter adapter : adapterCache.values()) {
            adapter.close();
        }
        adapterCache.clear();
        log.info("LoraAdapterService: cleared all cached adapters");
    }

    public Set<Path> getCachedAdapterPaths() {
        return Set.copyOf(adapterCache.keySet());
    }

    private Path resolveAdapterFile(Path adapterPath) throws IOException {
        if (Files.isRegularFile(adapterPath)) {
            String name = adapterPath.getFileName().toString().toLowerCase();
            if (name.endsWith(".safetensors") || name.endsWith(".safetensor")) {
                return adapterPath;
            }
            throw new IllegalArgumentException("File must be a .safetensors file: " + adapterPath);
        }

        if (Files.isDirectory(adapterPath)) {
            Path standard = adapterPath.resolve(DEFAULT_ADAPTER_FILE);
            if (Files.isRegularFile(standard)) return standard;

            Path peft = adapterPath.resolve(PEFT_ADAPTER_FILE);
            if (Files.isRegularFile(peft)) return peft;

            try (var stream = Files.list(adapterPath)) {
                Optional<Path> found = stream
                        .filter(p -> {
                            String name = p.getFileName().toString().toLowerCase();
                            return name.endsWith(".safetensors") && !name.contains("index");
                        })
                        .findFirst();
                if (found.isPresent()) return found.get();
            }
            throw new IllegalArgumentException("No adapter file found in " + adapterPath);
        }
        throw new IllegalArgumentException("Invalid adapter path: " + adapterPath);
    }

    private Map<String, TorchTensor> loadAdapterWeights(Path adapterFile) throws IOException {
        Map<String, TorchTensor> weights = new HashMap<>();
        try (SafetensorShardSession session = safetensorLoader.open(adapterFile)) {
            for (String name : session.tensorNames()) {
                if (isLoraTensor(name)) {
                    SafetensorTensor st = session.tensor(name);
                    weights.put(name, bridgeToLibTorch(st));
                }
            }
        } catch (Exception e) {
            throw new IOException("Failed to load adapter weights from " + adapterFile, e);
        }
        return weights;
    }

    private boolean isLoraTensor(String name) {
        return name.endsWith(".lora_A.weight") || name.endsWith(".lora_A")
                || name.endsWith(".lora_B.weight") || name.endsWith(".lora_B");
    }

    private TorchTensor bridgeToLibTorch(SafetensorTensor st) {
        float[] data = st.toFloatArray();
        long[] shape = st.shape();
        return TorchTensor.fromFloatArray(data, shape);
    }

    private AdapterConfig loadAdapterConfig(Path adapterDir) throws IOException {
        Path configFile = adapterDir.resolve(DEFAULT_CONFIG_FILE);
        if (!Files.isRegularFile(configFile)) {
            return createDefaultConfig();
        }

        try {
            JsonNode root = objectMapper.readTree(configFile.toFile());
            int rank = extractInt(root, "r").orElse(extractInt(root, "rank").orElse(16));
            float alpha = extractFloat(root, "lora_alpha").orElse((float) rank);
            float dropout = extractFloat(root, "lora_dropout").orElse(0.0f);
            String bias = extractString(root, "bias").orElse("none");
            String taskType = extractString(root, "task_type").orElse(null);
            boolean inferenceMode = extractBoolean(root, "inference_mode").orElse(true);
            String baseModelName = extractString(root, "base_model_name_or_path").orElse(null);
            List<String> targetModules = extractStringArray(root, "target_modules");
            Map<String, Object> properties = extractAdditionalProperties(root);

            return new AdapterConfig(rank, alpha, dropout, bias, targetModules,
                    taskType, inferenceMode, baseModelName, properties);
        } catch (IOException e) {
            return createDefaultConfig();
        }
    }

    private AdapterConfig createDefaultConfig() {
        return new AdapterConfig(16, 16.0f, 0.0f, "none", List.of(), null, true, null, Map.of());
    }

    private void validateRankConsistency(Map<String, TorchTensor> weights, int expectedRank) {
        for (Map.Entry<String, TorchTensor> entry : weights.entrySet()) {
            String name = entry.getKey();
            long[] shape = entry.getValue().shape();
            if (name.contains(".lora_A")) {
                if (shape[0] != expectedRank) {
                    log.warnf("LoraAdapterService: rank mismatch for %s: %d vs %d", name, shape[0], expectedRank);
                }
            } else if (name.contains(".lora_B")) {
                if (shape[1] != expectedRank) {
                    log.warnf("LoraAdapterService: rank mismatch for %s: %d vs %d", name, shape[1], expectedRank);
                }
            }
        }
    }
}
