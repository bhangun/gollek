package tech.kayys.gollek.onnx.runner;

import tech.kayys.gollek.onnx.binding.OnnxRuntimeBinding;

import java.util.Objects;

final class OnnxTextSessionResources implements AutoCloseable {

    private final OnnxTextSessionContract contract;
    private final OnnxTextPreparedRunNames preparedRunNames;
    private final int vocabSize;
    private final int kvHeads;
    private final int kvHeadSize;
    private final int kvElementType;
    private boolean closed;

    private OnnxTextSessionResources(
            OnnxTextSessionContract contract,
            OnnxTextPreparedRunNames preparedRunNames,
            int vocabSize,
            int kvHeads,
            int kvHeadSize,
            int kvElementType) {
        this.contract = Objects.requireNonNull(contract, "contract");
        this.preparedRunNames = preparedRunNames;
        if (vocabSize <= 0) {
            throw new IllegalArgumentException("vocabSize must be > 0");
        }
        if (kvHeads <= 0) {
            throw new IllegalArgumentException("kvHeads must be > 0");
        }
        if (kvHeadSize <= 0) {
            throw new IllegalArgumentException("kvHeadSize must be > 0");
        }
        this.vocabSize = vocabSize;
        this.kvHeads = kvHeads;
        this.kvHeadSize = kvHeadSize;
        this.kvElementType = kvElementType;
    }

    static OnnxTextSessionResources empty() {
        return new OnnxTextSessionResources(
                OnnxTextSessionContract.empty(),
                null,
                1,
                1,
                1,
                OnnxRuntimeBinding.ONNX_TENSOR_FLOAT16);
    }

    static OnnxTextSessionResources create(
            OnnxRuntimeBinding binding,
            OnnxTextSessionContract contract,
            int vocabSize,
            int kvHeads,
            int kvHeadSize,
            int kvElementType) {
        Objects.requireNonNull(binding, "binding");
        Objects.requireNonNull(contract, "contract");
        validate(vocabSize, kvHeads, kvHeadSize);
        return new OnnxTextSessionResources(
                contract,
                OnnxTextPreparedRunNames.create(binding, contract),
                vocabSize,
                kvHeads,
                kvHeadSize,
                kvElementType);
    }

    static OnnxTextSessionResources createForTest(
            OnnxTextSessionContract contract,
            OnnxTextPreparedRunNames preparedRunNames,
            int vocabSize,
            int kvHeads,
            int kvHeadSize,
            int kvElementType) {
        Objects.requireNonNull(preparedRunNames, "preparedRunNames");
        return new OnnxTextSessionResources(
                contract,
                preparedRunNames,
                vocabSize,
                kvHeads,
                kvHeadSize,
                kvElementType);
    }

    OnnxTextSessionContract contract() {
        return contract;
    }

    OnnxTextPreparedRunNames preparedRunNames() {
        ensureOpen();
        if (preparedRunNames == null) {
            throw new IllegalStateException("ONNX text prepared run names are not initialized");
        }
        return preparedRunNames;
    }

    int vocabSize() {
        return vocabSize;
    }

    int kvHeads() {
        return kvHeads;
    }

    int kvHeadSize() {
        return kvHeadSize;
    }

    int kvElementType() {
        return kvElementType;
    }

    @Override
    public void close() {
        if (closed) {
            return;
        }
        if (preparedRunNames != null) {
            preparedRunNames.close();
        }
        closed = true;
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("ONNX text session resources are already closed");
        }
    }

    private static void validate(int vocabSize, int kvHeads, int kvHeadSize) {
        if (vocabSize <= 0) {
            throw new IllegalArgumentException("vocabSize must be > 0");
        }
        if (kvHeads <= 0) {
            throw new IllegalArgumentException("kvHeads must be > 0");
        }
        if (kvHeadSize <= 0) {
            throw new IllegalArgumentException("kvHeadSize must be > 0");
        }
    }
}
