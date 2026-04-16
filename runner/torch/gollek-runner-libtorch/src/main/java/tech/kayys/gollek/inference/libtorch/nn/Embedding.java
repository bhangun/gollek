package tech.kayys.gollek.inference.libtorch.nn;

import tech.kayys.gollek.inference.libtorch.binding.LibTorchBinding;
import tech.kayys.gollek.inference.libtorch.core.Device;
import tech.kayys.gollek.inference.libtorch.core.ScalarType;
import tech.kayys.gollek.inference.libtorch.core.TorchTensor;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

/**
 * Embedding layer — a lookup table for fixed-size vectors.
 * <p>
 * Mirrors {@code libtorch::nn::Embedding}. Maps discrete tokens to dense
 * vectors of fixed size. Commonly used for word embeddings in NLP.
 * </p>
 * <p>
 * <h2>Usage Example</h2>
 * <pre>{@code
 * Embedding embedding = new Embedding(10000, 300);  // vocab_size=10k, dim=300
 * TorchTensor input = TorchTensor.fromLongArray(new long[]{1, 5, 42}, new long[]{3});
 * TorchTensor output = embedding.forward(input);  // shape: [3, 300]
 * }</pre>
 *
 * @see Module
 * @see TorchTensor
 * @since 1.0
 */
public class Embedding extends Module {

    private final long numEmbeddings;
    private final long embeddingDim;
    private final long paddingIdx;

    /**
     * Creates an Embedding layer.
     *
     * @param numEmbeddings total number of embeddings (vocabulary size)
     * @param embeddingDim  size of each embedding vector
     * @param paddingIdx    index to zero out (-1 if none)
     */
    public Embedding(long numEmbeddings, long embeddingDim, long paddingIdx) {
        this.numEmbeddings = numEmbeddings;
        this.embeddingDim = embeddingDim;
        this.paddingIdx = paddingIdx;

        // Initialize weight: [numEmbeddings, embeddingDim] with normal distribution
        TorchTensor weight = TorchTensor.randn(new long[] { numEmbeddings, embeddingDim },
                ScalarType.FLOAT, Device.CPU);
        registerParameter("weight", weight);
    }

    /**
     * Convenience constructor without padding.
     *
     * @param numEmbeddings total number of embeddings
     * @param embeddingDim  size of each embedding vector
     */
    public Embedding(long numEmbeddings, long embeddingDim) {
        this(numEmbeddings, embeddingDim, -1);
    }

    @Override
    public TorchTensor forward(TorchTensor input) {
        // Embedding lookup: weight[input]
        // Uses the native embedding function if available,
        // otherwise falls back to index_select
        TorchTensor weight = parameters.get("weight");

        LibTorchBinding binding = LibTorchBinding.getInstance();
        Arena opArena = Arena.ofConfined();
        try {
            var embeddingFn = binding.bindOptional(
                    LibTorchBinding.NN_EMBEDDING, LibTorchBinding.NN_EMBEDDING_DESC);

            if (embeddingFn.isPresent()) {
                MemorySegment result = (MemorySegment) embeddingFn.get().invoke(
                        weight.nativeHandle(), input.nativeHandle(),
                        paddingIdx, false, false);
                return new TorchTensor(result, opArena);
            }

            // Fallback: use index_select on dim 0
            var indexSelectFn = binding.bindOptional(
                    LibTorchBinding.TENSOR_INDEX_SELECT,
                    LibTorchBinding.TENSOR_INDEX_SELECT_DESC);

            if (indexSelectFn.isPresent()) {
                MemorySegment result = (MemorySegment) indexSelectFn.get().invoke(
                        weight.nativeHandle(), 0L, input.nativeHandle());
                return new TorchTensor(result, opArena);
            }

            opArena.close();
            throw new UnsupportedOperationException(
                    "Neither embedding nor index_select native functions are available");
        } catch (UnsupportedOperationException e) {
            throw e;
        } catch (Throwable t) {
            opArena.close();
            throw new RuntimeException("Embedding forward failed", t);
        }
    }

    @Override
    public String toString() {
        return String.format("Embedding(%d, %d%s)", numEmbeddings, embeddingDim,
                paddingIdx >= 0 ? ", padding_idx=" + paddingIdx : "");
    }
}
