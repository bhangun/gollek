package tech.kayys.gollek.safetensor.engine.generation;

import static tech.kayys.gollek.safetensor.engine.generation.GenerationTokenPolicy.buildBaseStopTokenIds;
import static tech.kayys.gollek.safetensor.engine.generation.GenerationTokenPolicy.buildTokenSamplingMasks;

import java.lang.foreign.Arena;
import java.nio.file.Path;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.engine.runtime.ModelRuntimeTraitsResolver;
import tech.kayys.alkhawarizm.safetensor.quantization.QuantizationEngine;
import tech.kayys.alkhawarizm.safetensor.spi.SafetensorEngine;
import tech.kayys.alkhawarizm.spi.model.ModelArchitecture;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;
import tech.kayys.alkhawarizm.spi.model.ModelRuntimeTraits;
import tech.kayys.gollek.tokenizer.spi.Tokenizer;

public class DirectLoadedModel implements SafetensorEngine.LoadedModel {
    private final Path path;
    private final Tokenizer tokenizer;
    private final String key;
    private final boolean quantized;
    private final QuantizationEngine.QuantStrategy quantStrategy;
    private final String quantCacheState;
    private final Path quantCachePath;
    private final ModelConfig config;
    private final ModelArchitecture architecture;
    private final ModelRuntimeTraits runtimeTraits;
    private final Set<Integer> baseStopTokenIds;
    private final TokenSamplingMasks baseTokenSamplingMasks;
    private final Map<String, AccelTensor> weights;
    private final Arena weightArena;

    public DirectLoadedModel(Path path, Map<String, AccelTensor> weights,
            Tokenizer tokenizer, String key,
            boolean quantized, QuantizationEngine.QuantStrategy quantStrategy,
            String quantCacheState, Path quantCachePath,
            ModelConfig config, ModelArchitecture architecture, ModelRuntimeTraits runtimeTraits,
            Arena weightArena) {
        this.path = path;
        this.weights = weights;
        this.tokenizer = tokenizer;
        this.key = key;
        this.quantized = quantized;
        this.quantStrategy = quantStrategy;
        this.quantCacheState = quantCacheState;
        this.quantCachePath = quantCachePath;
        this.config = config != null ? config : new ModelConfig();
        this.architecture = Objects.requireNonNull(architecture, "architecture must not be null");
        this.runtimeTraits = ModelRuntimeTraitsResolver.resolve(this.architecture, this.config, runtimeTraits);
        this.baseStopTokenIds = buildBaseStopTokenIds(tokenizer, this.config);
        this.baseTokenSamplingMasks = buildTokenSamplingMasks(tokenizer, this.baseStopTokenIds,
                this.config.getVocabSize(), this.runtimeTraits);
        this.weightArena = weightArena;
    }

    @Override
    public Path path() {
        return path;
    }

    @Override
    public Map<String, AccelTensor> weights() {
        return weights;
    }

    @Override
    public Tokenizer tokenizer() {
        return tokenizer;
    }

    @Override
    public String key() {
        return key;
    }

    @Override
    public boolean isQuantized() {
        return quantized;
    }

    @Override
    public ModelConfig config() {
        return config;
    }

    public String arch() {
        return config.getPrimaryArchitecture();
    }

    public ModelArchitecture architecture() {
        return architecture;
    }

    @Override
    public ModelRuntimeTraits runtimeTraits() {
        return runtimeTraits;
    }

    Set<Integer> baseStopTokenIds() {
        return baseStopTokenIds;
    }

    TokenSamplingMasks baseTokenSamplingMasks() {
        return baseTokenSamplingMasks;
    }

    public QuantizationEngine.QuantStrategy getQuantStrategy() {
        return quantStrategy;
    }

    public String getQuantCacheState() {
        return quantCacheState;
    }

    public Path getQuantCachePath() {
        return quantCachePath;
    }

    void closeWeightArena() {
        if (weightArena == null) {
            return;
        }
        weightArena.close();
    }
}
