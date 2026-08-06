package tech.kayys.gollek.onnx.runner;

import tech.kayys.gollek.onnx.binding.OnnxRuntimeBinding;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Objects;
import java.util.function.Consumer;

final class OnnxTextRunWorkspace implements AutoCloseable {

    private final Arena arena;
    private final int capacity;
    private final boolean hasKvInputs;
    private final OnnxPastKvState pastKvState;
    private final OnnxTextInputAssembler inputAssembler;
    private final OnnxPreparedRun preparedRun;
    private final OnnxRunOutputValues outputValues;
    private final OnnxRunOutputLifecycle outputLifecycle;
    private final OnnxLogitsSelector logitsSelector;
    private final OnnxRepeatedInputValue emptyPastKvInput;
    private final Consumer<MemorySegment> releaseValue;
    private final OnnxTokenHistory tokenHistory;
    private final OnnxGeneratedTokens generatedTokens;
    private final OnnxTextDecodeStep decodeStep;
    private boolean requestActive;
    private boolean destroyed;

    private OnnxTextRunWorkspace(
            Arena arena,
            int capacity,
            boolean hasKvInputs,
            OnnxPastKvState pastKvState,
            OnnxTextInputAssembler inputAssembler,
            OnnxPreparedRun preparedRun,
            OnnxRunOutputValues outputValues,
            OnnxRunOutputLifecycle outputLifecycle,
            OnnxLogitsSelector logitsSelector,
            OnnxRepeatedInputValue emptyPastKvInput,
            Consumer<MemorySegment> releaseValue) {
        this.arena = arena;
        if (capacity < 0) {
            throw new IllegalArgumentException("capacity must be >= 0");
        }
        this.capacity = capacity;
        this.hasKvInputs = hasKvInputs;
        this.pastKvState = Objects.requireNonNull(pastKvState, "pastKvState");
        this.inputAssembler = Objects.requireNonNull(inputAssembler, "inputAssembler");
        this.preparedRun = Objects.requireNonNull(preparedRun, "preparedRun");
        this.outputValues = Objects.requireNonNull(outputValues, "outputValues");
        this.outputLifecycle = Objects.requireNonNull(outputLifecycle, "outputLifecycle");
        this.logitsSelector = Objects.requireNonNull(logitsSelector, "logitsSelector");
        this.emptyPastKvInput = Objects.requireNonNull(emptyPastKvInput, "emptyPastKvInput");
        this.releaseValue = Objects.requireNonNull(releaseValue, "releaseValue");
        this.tokenHistory = OnnxTokenHistory.allocate(capacity);
        this.generatedTokens = OnnxGeneratedTokens.allocate(capacity);
        this.decodeStep = OnnxTextDecodeStep.reusable();
    }

    static OnnxTextRunWorkspace create(
            OnnxRuntimeBinding binding,
            MemorySegment session,
            MemorySegment memInfo,
            OnnxTextSessionResources resources,
            int promptLength,
            int maxTokens,
            int inputTensorCacheEntries) {
        Objects.requireNonNull(binding, "binding");
        Objects.requireNonNull(session, "session");
        Objects.requireNonNull(memInfo, "memInfo");
        Objects.requireNonNull(resources, "resources");

        Arena arena = Arena.ofShared();
        try {
            return createWithArena(
                    binding,
                    session,
                    arena,
                    memInfo,
                    resources,
                    promptLength,
                    maxTokens,
                    inputTensorCacheEntries,
                    true);
        } catch (RuntimeException | Error e) {
            arena.close();
            throw e;
        }
    }

    private static OnnxTextRunWorkspace createWithArena(
            OnnxRuntimeBinding binding,
            MemorySegment session,
            Arena arena,
            MemorySegment memInfo,
            OnnxTextSessionResources resources,
            int promptLength,
            int maxTokens,
            int inputTensorCacheEntries,
            boolean ownsArena) {
        Objects.requireNonNull(arena, "arena");

        OnnxTextSessionContract contract = resources.contract();
        OnnxTextPreparedRunNames preparedNames = resources.preparedRunNames();
        boolean hasKvInputs = contract.hasKvInputs();
        String[] pastKvNames = contract.pastKvInputNames();
        int capacity = capacityFor(promptLength, maxTokens);
        Consumer<MemorySegment> releaseValue = binding::releaseValue;
        OnnxPastKvState pastKvState = OnnxPastKvState.allocate(hasKvInputs ? pastKvNames.length : 0);
        MemorySegment inputValuePtrs = binding.allocatePointerArray(arena, preparedNames.inputCount());
        MemorySegment outputValuePtrs = binding.allocatePointerArray(arena, preparedNames.outputCount());
        MemorySegment tensorValueOutPtr = binding.allocateValuePointer(arena);
        OnnxRunOutputValues outputValues = OnnxRunOutputValues.allocate(preparedNames.outputCount());
        OnnxRunOutputLifecycle outputLifecycle = OnnxRunOutputLifecycle.create(
                outputValues,
                pastKvState,
                hasKvInputs,
                releaseValue);
        OnnxPreparedRun preparedRun = OnnxPreparedRun.create(
                binding,
                session,
                preparedNames.inputNamePointers(),
                inputValuePtrs,
                preparedNames.outputNamePointers(),
                outputValuePtrs,
                outputValues);
        OnnxLogitsSelector logitsSelector = OnnxLogitsSelector.create(binding, resources.vocabSize(), hasKvInputs);
        MemorySegment emptyPastKvShape = MemorySegment.NULL;
        if (hasKvInputs) {
            emptyPastKvShape = binding.allocateShapeBuffer(arena, 4);
            binding.writeShape4d(emptyPastKvShape, 1, resources.kvHeads(), 0, resources.kvHeadSize());
        }
        MemorySegment finalEmptyPastKvShape = emptyPastKvShape;
        OnnxRepeatedInputValue emptyPastKvInput = OnnxRepeatedInputValue.create(
                hasKvInputs ? pastKvNames.length : 0,
                () -> binding.createTensorWithPreparedShape(
                        memInfo,
                        MemorySegment.NULL,
                        finalEmptyPastKvShape,
                        4,
                        resources.kvElementType(),
                        tensorValueOutPtr),
                releaseValue);
        OnnxTextInputAssembler inputAssembler = OnnxTextInputAssembler.create(
                binding,
                arena,
                memInfo,
                tensorValueOutPtr,
                preparedNames.inputCount(),
                promptLength,
                maxTokens,
                contract.hasPositionIds(),
                hasKvInputs,
                pastKvNames.length,
                emptyPastKvInput,
                inputTensorCacheEntries,
                releaseValue);
        return new OnnxTextRunWorkspace(
                ownsArena ? arena : null,
                capacity,
                hasKvInputs,
                pastKvState,
                inputAssembler,
                preparedRun,
                outputValues,
                outputLifecycle,
                logitsSelector,
                emptyPastKvInput,
                releaseValue);
    }

    static OnnxTextRunWorkspace createForTest(
            int capacity,
            boolean hasKvInputs,
            OnnxPastKvState pastKvState,
            OnnxTextInputAssembler inputAssembler,
            OnnxPreparedRun preparedRun,
            OnnxRunOutputValues outputValues,
            OnnxRunOutputLifecycle outputLifecycle,
            OnnxLogitsSelector logitsSelector,
            OnnxRepeatedInputValue emptyPastKvInput,
            Consumer<MemorySegment> releaseValue) {
        return new OnnxTextRunWorkspace(
                null,
                capacity,
                hasKvInputs,
                pastKvState,
                inputAssembler,
                preparedRun,
                outputValues,
                outputLifecycle,
                logitsSelector,
                emptyPastKvInput,
                releaseValue);
    }

    static int capacityFor(int promptLength, int maxTokens) {
        if (promptLength <= 0) {
            throw new IllegalArgumentException("promptLength must be > 0");
        }
        if (maxTokens <= 0) {
            throw new IllegalArgumentException("maxTokens must be > 0");
        }
        return Math.max(1, promptLength + Math.max(1, maxTokens));
    }

    boolean canServe(int promptLength, int maxTokens) {
        return !destroyed && capacity >= capacityFor(promptLength, maxTokens);
    }

    int capacity() {
        return capacity;
    }

    void beginRequest() {
        ensureNotDestroyed();
        if (requestActive) {
            throw new IllegalStateException("ONNX text run workspace is already active");
        }
        inputAssembler.resetForRequest();
        requestActive = true;
    }

    boolean hasKvInputs() {
        return hasKvInputs;
    }

    OnnxInputTensorCacheStats inputTensorCacheStats() {
        return inputAssembler.cacheStats();
    }

    OnnxTokenHistory resetTokenHistory(int[] prompt, int expectedGeneratedTokens) {
        ensureActive();
        return tokenHistory.resetFrom(prompt, expectedGeneratedTokens);
    }

    OnnxGeneratedTokens resetGeneratedTokens(int expectedGeneratedTokens) {
        ensureActive();
        return generatedTokens.reset(expectedGeneratedTokens);
    }

    OnnxTextDecodeStep planDecodeStep(int promptLength, int tokenHistorySize, long consumedTokens) {
        ensureActive();
        return decodeStep.update(
                hasKvInputs,
                pastKvState.hasCurrent(),
                promptLength,
                tokenHistorySize,
                consumedTokens);
    }

    OnnxRunInputValues assembleInputs(
            OnnxTokenHistory tokenIds,
            int promptLength,
            OnnxTextDecodeStep decodeStep) {
        ensureActive();
        return inputAssembler.assemble(tokenIds, promptLength, decodeStep, pastKvState);
    }

    OnnxRunOutputValues execute(
            OnnxRunInputValues inputValues,
            OnnxInferenceProfile profile,
            long inputPrepareStart,
            boolean prefill) {
        ensureActive();
        return preparedRun.execute(inputValues, profile, inputPrepareStart, prefill);
    }

    int selectNextToken(long sequenceLength) {
        ensureActive();
        return logitsSelector.selectNextToken(outputValues.logits(), sequenceLength);
    }

    void releaseOwnedInputs(OnnxRunInputValues inputValues) {
        ensureActive();
        Objects.requireNonNull(inputValues, "inputValues").releaseOwned(releaseValue);
    }

    void capturePresentAndReleaseLogits() {
        ensureActive();
        outputLifecycle.capturePresentAndReleaseLogits();
    }

    void finishRequest() {
        if (!requestActive) {
            return;
        }
        outputLifecycle.releaseUncapturedAndCurrentPastKv();
        requestActive = false;
    }

    @Override
    public void close() {
        if (destroyed) {
            return;
        }
        finishRequest();
        inputAssembler.releaseCachedInputs(releaseValue);
        emptyPastKvInput.release();
        if (arena != null) {
            arena.close();
        }
        destroyed = true;
    }

    private void ensureActive() {
        ensureNotDestroyed();
        if (!requestActive) {
            throw new IllegalStateException("ONNX text run workspace is not active");
        }
    }

    private void ensureNotDestroyed() {
        if (destroyed) {
            throw new IllegalStateException("ONNX text run workspace is already closed");
        }
    }
}
