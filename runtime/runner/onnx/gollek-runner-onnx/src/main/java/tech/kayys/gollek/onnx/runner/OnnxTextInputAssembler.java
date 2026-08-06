package tech.kayys.gollek.onnx.runner;

import tech.kayys.gollek.onnx.binding.OnnxRuntimeBinding;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Objects;
import java.util.function.Consumer;

final class OnnxTextInputAssembler {

    private static final int BATCH_SIZE = 1;
    static final int DEFAULT_INPUT_TENSOR_CACHE_ENTRIES = 32;

    private final Ops ops;
    private final MemorySegment memInfo;
    private final MemorySegment rank2Shape;
    private final MemorySegment tensorValueOutPtr;
    private final OnnxRunInputValues inputValues;
    private final OnnxInputIdsScratch inputIdsScratch;
    private final OnnxAttentionMaskScratch attentionMaskScratch;
    private final OnnxPositionIdsScratch positionIdsScratch;
    private final OnnxLengthTensorCache prefixInputIdsTensorCache;
    private final OnnxLengthTensorCache attentionMaskTensorCache;
    private final OnnxPositionIdsTensorCache positionIdsTensorCache;
    private final boolean hasKvInputs;
    private final int pastKvInputCount;
    private final OnnxRepeatedInputValue emptyPastKvInput;
    private final Consumer<MemorySegment> cacheEvictionReleaser;
    private MemorySegment cachedScalarInputIdsTensor;
    private MemorySegment cachedScalarPositionIdsTensor;
    private int scalarInputIdsCacheHits;
    private int scalarInputIdsCacheMisses;
    private int scalarPositionIdsCacheHits;
    private int scalarPositionIdsCacheMisses;

    private OnnxTextInputAssembler(
            Ops ops,
            MemorySegment memInfo,
            MemorySegment rank2Shape,
            MemorySegment tensorValueOutPtr,
            OnnxRunInputValues inputValues,
            OnnxInputIdsScratch inputIdsScratch,
            OnnxAttentionMaskScratch attentionMaskScratch,
            OnnxPositionIdsScratch positionIdsScratch,
            boolean hasKvInputs,
            int pastKvInputCount,
            OnnxRepeatedInputValue emptyPastKvInput,
            int inputTensorCacheEntries,
            Consumer<MemorySegment> cacheEvictionReleaser) {
        this.ops = Objects.requireNonNull(ops, "ops");
        this.memInfo = Objects.requireNonNull(memInfo, "memInfo");
        this.rank2Shape = Objects.requireNonNull(rank2Shape, "rank2Shape");
        this.tensorValueOutPtr = Objects.requireNonNull(tensorValueOutPtr, "tensorValueOutPtr");
        this.inputValues = Objects.requireNonNull(inputValues, "inputValues");
        this.inputIdsScratch = Objects.requireNonNull(inputIdsScratch, "inputIdsScratch");
        this.attentionMaskScratch = Objects.requireNonNull(attentionMaskScratch, "attentionMaskScratch");
        this.positionIdsScratch = positionIdsScratch;
        this.prefixInputIdsTensorCache = new OnnxLengthTensorCache(inputTensorCacheEntries);
        this.attentionMaskTensorCache = new OnnxLengthTensorCache(inputTensorCacheEntries);
        this.positionIdsTensorCache = new OnnxPositionIdsTensorCache(inputTensorCacheEntries);
        this.hasKvInputs = hasKvInputs;
        if (pastKvInputCount < 0) {
            throw new IllegalArgumentException("pastKvInputCount must be >= 0");
        }
        this.pastKvInputCount = pastKvInputCount;
        if (hasKvInputs) {
            this.emptyPastKvInput = Objects.requireNonNull(emptyPastKvInput, "emptyPastKvInput");
        } else {
            this.emptyPastKvInput = emptyPastKvInput;
        }
        this.cacheEvictionReleaser = Objects.requireNonNull(cacheEvictionReleaser, "cacheEvictionReleaser");
    }

    static OnnxTextInputAssembler create(
            OnnxRuntimeBinding binding,
            Arena arena,
            MemorySegment memInfo,
            MemorySegment tensorValueOutPtr,
            int inputCount,
            int promptLength,
            int maxTokens,
            boolean hasPositionIds,
            boolean hasKvInputs,
            int pastKvInputCount,
            OnnxRepeatedInputValue emptyPastKvInput,
            int inputTensorCacheEntries,
            Consumer<MemorySegment> cacheEvictionReleaser) {
        Objects.requireNonNull(binding, "binding");
        Objects.requireNonNull(arena, "arena");
        int capacity = Math.max(1, promptLength + Math.max(1, maxTokens));
        int ownedInputCount = hasPositionIds ? 3 : 2;
        return new OnnxTextInputAssembler(
                new BindingOps(binding),
                memInfo,
                binding.allocateShapeBuffer(arena, 2),
                tensorValueOutPtr,
                OnnxRunInputValues.allocate(inputCount, ownedInputCount),
                OnnxInputIdsScratch.allocate(arena, capacity),
                OnnxAttentionMaskScratch.allocate(arena, capacity),
                hasPositionIds ? OnnxPositionIdsScratch.allocate(arena, capacity) : null,
                hasKvInputs,
                pastKvInputCount,
                emptyPastKvInput,
                inputTensorCacheEntries,
                cacheEvictionReleaser);
    }

    static OnnxTextInputAssembler createForTest(
            Ops ops,
            MemorySegment memInfo,
            MemorySegment rank2Shape,
            MemorySegment tensorValueOutPtr,
            OnnxRunInputValues inputValues,
            OnnxInputIdsScratch inputIdsScratch,
            OnnxAttentionMaskScratch attentionMaskScratch,
            OnnxPositionIdsScratch positionIdsScratch,
            boolean hasKvInputs,
            int pastKvInputCount,
            OnnxRepeatedInputValue emptyPastKvInput) {
        return createForTest(
                ops,
                memInfo,
                rank2Shape,
                tensorValueOutPtr,
                inputValues,
                inputIdsScratch,
                attentionMaskScratch,
                positionIdsScratch,
                hasKvInputs,
                pastKvInputCount,
                emptyPastKvInput,
                DEFAULT_INPUT_TENSOR_CACHE_ENTRIES,
                ignored -> {
                });
    }

    static OnnxTextInputAssembler createForTest(
            Ops ops,
            MemorySegment memInfo,
            MemorySegment rank2Shape,
            MemorySegment tensorValueOutPtr,
            OnnxRunInputValues inputValues,
            OnnxInputIdsScratch inputIdsScratch,
            OnnxAttentionMaskScratch attentionMaskScratch,
            OnnxPositionIdsScratch positionIdsScratch,
            boolean hasKvInputs,
            int pastKvInputCount,
            OnnxRepeatedInputValue emptyPastKvInput,
            int inputTensorCacheEntries,
            Consumer<MemorySegment> cacheEvictionReleaser) {
        return new OnnxTextInputAssembler(
                ops,
                memInfo,
                rank2Shape,
                tensorValueOutPtr,
                inputValues,
                inputIdsScratch,
                attentionMaskScratch,
                positionIdsScratch,
                hasKvInputs,
                pastKvInputCount,
                emptyPastKvInput,
                inputTensorCacheEntries,
                cacheEvictionReleaser);
    }

    OnnxRunInputValues assemble(
            OnnxTokenHistory tokenIds,
            int promptLength,
            OnnxTextDecodeStep decodeStep,
            OnnxPastKvState pastKvState) {
        Objects.requireNonNull(tokenIds, "tokenIds");
        Objects.requireNonNull(decodeStep, "decodeStep");
        if (hasKvInputs) {
            Objects.requireNonNull(pastKvState, "pastKvState");
        }

        long seqLen = decodeStep.sequenceLength();
        inputValues.reset();
        addInputIdsTensor(tokenIds, promptLength, decodeStep, seqLen);
        addAttentionMaskTensor(decodeStep);
        if (positionIdsScratch != null) {
            addPositionIdsTensor(decodeStep, seqLen);
        }

        if (hasKvInputs && pastKvState.hasCurrent()) {
            inputValues.addBorrowedAllUnchecked(pastKvState.current(), pastKvInputCount);
        } else if (hasKvInputs) {
            inputValues.addRepeated(emptyPastKvInput);
        }
        return inputValues;
    }

    void resetForRequest() {
        inputValues.reset();
        inputIdsScratch.reset();
        prefixInputIdsTensorCache.resetStats();
        attentionMaskTensorCache.resetStats();
        positionIdsTensorCache.resetStats();
        scalarInputIdsCacheHits = 0;
        scalarInputIdsCacheMisses = 0;
        scalarPositionIdsCacheHits = 0;
        scalarPositionIdsCacheMisses = 0;
    }

    void releaseCachedInputs(Consumer<MemorySegment> releaser) {
        Objects.requireNonNull(releaser, "releaser");
        cachedScalarInputIdsTensor = releaseCached(releaser, cachedScalarInputIdsTensor);
        cachedScalarPositionIdsTensor = releaseCached(releaser, cachedScalarPositionIdsTensor);
        prefixInputIdsTensorCache.releaseAll(releaser);
        attentionMaskTensorCache.releaseAll(releaser);
        positionIdsTensorCache.releaseAll(releaser);
    }

    OnnxInputTensorCacheStats cacheStats() {
        return new OnnxInputTensorCacheStats(
                scalarInputIdsCacheHits,
                scalarInputIdsCacheMisses,
                scalarPositionIdsCacheHits,
                scalarPositionIdsCacheMisses,
                attentionMaskTensorCache.hits(),
                attentionMaskTensorCache.misses(),
                prefixInputIdsTensorCache.hits(),
                prefixInputIdsTensorCache.misses(),
                positionIdsTensorCache.hits(),
                positionIdsTensorCache.misses(),
                attentionMaskTensorCache.evictions()
                        + prefixInputIdsTensorCache.evictions()
                        + positionIdsTensorCache.evictions());
    }

    private void addInputIdsTensor(
            OnnxTokenHistory tokenIds,
            int promptLength,
            OnnxTextDecodeStep decodeStep,
            long seqLen) {
        OnnxTensorDataView ids64;
        if (decodeStep.prefill()) {
            ids64 = inputIdsScratch.prefixView(tokenIds, promptLength);
        } else if (hasKvInputs) {
            ids64 = inputIdsScratch.lastView(tokenIds);
            if (cachedScalarInputIdsTensor == null) {
                scalarInputIdsCacheMisses++;
                cachedScalarInputIdsTensor = createCachedScalarTensor(ids64);
            } else {
                scalarInputIdsCacheHits++;
            }
            inputValues.addBorrowed(cachedScalarInputIdsTensor);
            return;
        } else {
            ids64 = inputIdsScratch.prefixView(tokenIds, tokenIds.size());
        }
        MemorySegment tensor = prefixInputIdsTensorCache.get(seqLen);
        if (tensor != null) {
            inputValues.addBorrowed(tensor);
            return;
        }
        tensor = createInt64Tensor(ids64, seqLen);
        addCachedTensor(prefixInputIdsTensorCache.retain(seqLen, tensor, cacheEvictionReleaser), tensor);
    }

    private void addAttentionMaskTensor(OnnxTextDecodeStep decodeStep) {
        long attentionLength = decodeStep.attentionLength();
        MemorySegment tensor = attentionMaskTensorCache.get(attentionLength);
        if (tensor != null) {
            inputValues.addBorrowed(tensor);
            return;
        }
        OnnxTensorDataView mask = attentionMaskScratch.onesView(attentionLength);
        tensor = createInt64Tensor(mask, attentionLength);
        addCachedTensor(attentionMaskTensorCache.retain(attentionLength, tensor, cacheEvictionReleaser), tensor);
    }

    private void addPositionIdsTensor(OnnxTextDecodeStep decodeStep, long seqLen) {
        long positionStart = decodeStep.positionStart();
        OnnxTensorDataView positions = positionIdsScratch.positionsView(positionStart, seqLen);
        if (hasKvInputs && !decodeStep.prefill() && seqLen == 1L) {
            if (cachedScalarPositionIdsTensor == null) {
                scalarPositionIdsCacheMisses++;
                cachedScalarPositionIdsTensor = createCachedScalarTensor(positions);
            } else {
                scalarPositionIdsCacheHits++;
            }
            inputValues.addBorrowed(cachedScalarPositionIdsTensor);
            return;
        }
        if (seqLen > 1L) {
            MemorySegment tensor = positionIdsTensorCache.get(positionStart, seqLen);
            if (tensor != null) {
                inputValues.addBorrowed(tensor);
                return;
            }
            tensor = createInt64Tensor(positions, seqLen);
            addCachedTensor(
                    positionIdsTensorCache.retain(positionStart, seqLen, tensor, cacheEvictionReleaser),
                    tensor);
            return;
        }
        inputValues.addOwned(createInt64Tensor(positions, seqLen));
    }

    private MemorySegment createInt64Tensor(OnnxTensorDataView view, long sequenceLength) {
        ops.writeShape2d(rank2Shape, BATCH_SIZE, sequenceLength);
        return ops.createInt64Tensor(memInfo, view, rank2Shape, tensorValueOutPtr);
    }

    private void addCachedTensor(boolean retained, MemorySegment tensor) {
        if (retained) {
            inputValues.addBorrowed(tensor);
        } else {
            inputValues.addOwned(tensor);
        }
    }

    private MemorySegment createCachedScalarTensor(OnnxTensorDataView view) {
        if (view.byteLength() != Long.BYTES) {
            throw new IllegalArgumentException("Cached scalar tensor requires one int64 element, found bytes="
                    + view.byteLength());
        }
        return createInt64Tensor(view, 1L);
    }

    private MemorySegment releaseCached(Consumer<MemorySegment> releaser, MemorySegment value) {
        if (value != null) {
            releaser.accept(value);
        }
        return null;
    }

    interface Ops {
        void writeShape2d(MemorySegment shapeBuffer, long firstDim, long secondDim);

        MemorySegment createInt64Tensor(
                MemorySegment memInfo,
                OnnxTensorDataView view,
                MemorySegment shapeBuffer,
                MemorySegment valueOutPointer);
    }

    private record BindingOps(OnnxRuntimeBinding binding) implements Ops {
        private BindingOps {
            Objects.requireNonNull(binding, "binding");
        }

        @Override
        public void writeShape2d(MemorySegment shapeBuffer, long firstDim, long secondDim) {
            binding.writeShape2d(shapeBuffer, firstDim, secondDim);
        }

        @Override
        public MemorySegment createInt64Tensor(
                MemorySegment memInfo,
                OnnxTensorDataView view,
                MemorySegment shapeBuffer,
                MemorySegment valueOutPointer) {
            return binding.createTensorWithPreparedShape(
                    memInfo,
                    view.data(),
                    view.byteLength(),
                    shapeBuffer,
                    2,
                    OnnxRuntimeBinding.ONNX_TENSOR_INT64,
                    valueOutPointer);
        }
    }
}
