package tech.kayys.gollek.onnx.runner;

import ai.onnxruntime.NodeInfo;
import ai.onnxruntime.OnnxJavaType;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxTensorLike;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.TensorInfo;
import ai.onnxruntime.ValueInfo;
import tech.kayys.gollek.tokenizer.runtime.TokenizerFactory;
import tech.kayys.gollek.tokenizer.spi.DecodeOptions;
import tech.kayys.gollek.tokenizer.spi.EncodeOptions;
import tech.kayys.gollek.tokenizer.spi.Tokenizer;

import java.io.IOException;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public final class PaddleOcrVlOnnxProbe {
    private static final Pattern LOC_TOKEN_PATTERN = Pattern.compile("<\\|LOC_(\\d{1,4})\\|>");

    private PaddleOcrVlOnnxProbe() {
    }

    public static VisionEncoderProbeResult runVisionEncoder(
            Path modelDir,
            Path imagePath,
            String requestedVariant,
            int threads) {
        try {
            return runVisionEncoderChecked(modelDir, imagePath, requestedVariant, threads);
        } catch (OrtException e) {
            throw new IllegalStateException("ORT vision_encoder execution failed: " + e.getMessage(), e);
        }
    }

    public static GraphIoProbeResult inspectGraph(Path graphPath, int threads) {
        Objects.requireNonNull(graphPath, "graphPath");
        if (!Files.isRegularFile(graphPath)) {
            throw new IllegalArgumentException("ONNX graph not found: " + graphPath);
        }
        OrtEnvironment env = OrtEnvironment.getEnvironment("gollek-paddleocr-vl");
        long loadStarted = System.nanoTime();
        try (OrtSession.SessionOptions options = new OrtSession.SessionOptions()) {
            options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
            options.setIntraOpNumThreads(Math.max(1, threads));
            options.setInterOpNumThreads(1);
            try (OrtSession session = env.createSession(graphPath.toAbsolutePath().toString(), options)) {
                return new GraphIoProbeResult(
                        graphPath.toAbsolutePath().normalize(),
                        Duration.ofNanos(System.nanoTime() - loadStarted),
                        describeInfo(session.getInputInfo()),
                        describeInfo(session.getOutputInfo()));
            }
        } catch (OrtException e) {
            throw new IllegalStateException("ORT graph inspection failed for "
                    + graphPath + ": " + e.getMessage(), e);
        }
    }

    public static DecoderPrefillProbeResult runDecoderPrefill(
            Path modelDir,
            Path imagePath,
            String requestedVariant,
            int imageTokenLimit,
            int threads) {
        try {
            return runDecoderPrefillChecked(modelDir, imagePath, requestedVariant, imageTokenLimit, threads);
        } catch (OrtException e) {
            throw new IllegalStateException("ORT decoder prefill probe failed: " + e.getMessage(), e);
        }
    }

    public static PromptDecodeProbeResult runPromptDecode(
            Path modelDir,
            Path imagePath,
            String userPrompt,
            String requestedVariant,
            int imageTokenLimit,
            int maxNewTokens,
            int threads) {
        try {
            return runPromptDecodeChecked(
                    modelDir,
                    imagePath,
                    userPrompt,
                    requestedVariant,
                    imageTokenLimit,
                    maxNewTokens,
                    threads);
        } catch (OrtException e) {
            throw new IllegalStateException("ORT prompt decode probe failed: " + e.getMessage(), e);
        } catch (IOException e) {
            throw new IllegalStateException("PaddleOCR-VL tokenizer load failed: " + e.getMessage(), e);
        }
    }

    public static OcrPostProcessResult postProcessOcrText(
            String decodedText,
            PaddleOcrVlOnnxPlanner.ImagePlan image) {
        String raw = decodedText == null ? "" : decodedText;
        List<Integer> locations = extractLocationTokens(raw);
        List<LocationBox> boxes = locationBoxes(locations, image);
        List<OcrTextRegion> regions = textRegions(raw, boxes);
        String textWithoutLocations = LOC_TOKEN_PATTERN.matcher(raw).replaceAll(" ")
                .replaceAll("\\s+", " ")
                .trim();
        String displayText = ocrDisplayText(raw, textWithoutLocations, locations, boxes, regions);
        return new OcrPostProcessResult(
                raw,
                displayText,
                textWithoutLocations,
                locations,
                boxes,
                regions);
    }

    private static VisionEncoderProbeResult runVisionEncoderChecked(
            Path modelDir,
            Path imagePath,
            String requestedVariant,
            int threads) throws OrtException {
        Objects.requireNonNull(modelDir, "modelDir");
        Objects.requireNonNull(imagePath, "imagePath");

        PaddleOcrVlOnnxPlanner.Plan plan =
                PaddleOcrVlOnnxPlanner.plan(modelDir, List.of(imagePath), requestedVariant);
        if (plan.images().isEmpty()) {
            throw new IllegalArgumentException("vision encoder probe needs one input image");
        }
        Path visionGraph = plan.graphs().visionEncoder();
        if (visionGraph == null || !Files.isRegularFile(visionGraph)) {
            throw new IllegalArgumentException("vision_encoder graph not found: " + visionGraph);
        }

        PaddleOcrVlOnnxPlanner.ImageTensor imageTensor =
                PaddleOcrVlOnnxPlanner.preprocessImage(imagePath, plan.processor());
        OrtEnvironment env = OrtEnvironment.getEnvironment("gollek-paddleocr-vl");

        long loadStarted = System.nanoTime();
        try (OrtSession.SessionOptions options = new OrtSession.SessionOptions()) {
            options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
            options.setIntraOpNumThreads(Math.max(1, threads));
            options.setInterOpNumThreads(1);
            try (OrtSession session = env.createSession(visionGraph.toAbsolutePath().toString(), options)) {
                Duration loadDuration = Duration.ofNanos(System.nanoTime() - loadStarted);
                List<IoInfo> inputs = describeInfo(session.getInputInfo());
                List<IoInfo> declaredOutputs = describeInfo(session.getOutputInfo());
                String pixelInputName = matchingInputName(inputs, "pixel_values");
                String gridInputName = matchingInputName(inputs, "image_grid_thw");
                if (pixelInputName == null || gridInputName == null) {
                    throw new IllegalArgumentException("vision_encoder inputs do not include pixel_values and image_grid_thw: "
                            + inputNames(inputs));
                }

                try (OnnxTensor pixelValues = OnnxTensor.createTensor(
                        env,
                        FloatBuffer.wrap(imageTensor.pixelValues()),
                        imageTensor.pixelValuesShape());
                     OnnxTensor imageGridThw = createGridTensor(
                             env,
                             imageTensor,
                             inputType(inputs, gridInputName))) {
                    Map<String, OnnxTensorLike> feeds = new LinkedHashMap<>();
                    feeds.put(pixelInputName, pixelValues);
                    feeds.put(gridInputName, imageGridThw);

                    long runStarted = System.nanoTime();
                    try (OrtSession.Result result = session.run(feeds)) {
                        Duration runDuration = Duration.ofNanos(System.nanoTime() - runStarted);
                        return new VisionEncoderProbeResult(
                                plan,
                                imageTensor,
                                visionGraph.toAbsolutePath().normalize(),
                                loadDuration,
                                runDuration,
                                inputs,
                                declaredOutputs,
                                describeResult(result));
                    }
                }
            }
        }
    }

    private static DecoderPrefillProbeResult runDecoderPrefillChecked(
            Path modelDir,
            Path imagePath,
            String requestedVariant,
            int imageTokenLimit,
            int threads) throws OrtException {
        VisionEmbedsResult vision = runVisionEncoderWithEmbeds(modelDir, imagePath, requestedVariant, threads);
        GraphTensorResult textEmbeds = runEmbedding(vision.plan().graphs().embedding(), threads, new long[] { 1L, 2L });

        long imageTokenCount = vision.imageEmbedsShape().length == 0 ? 0 : vision.imageEmbedsShape()[0];
        if (imageTokenCount <= 0) {
            throw new IllegalArgumentException("vision_encoder produced no image embeddings");
        }
        int hiddenSize = hiddenSize(vision.imageEmbedsShape(), textEmbeds.shape());
        int usedImageTokens = imageTokenLimit <= 0
                ? Math.toIntExact(imageTokenCount)
                : (int) Math.min(imageTokenCount, imageTokenLimit);
        int textTokens = textEmbeds.shape().length >= 2 ? Math.toIntExact(textEmbeds.shape()[1]) : 2;
        if (textTokens < 2) {
            throw new IllegalArgumentException("embedding graph returned fewer than two token embeddings");
        }
        int sequenceLength = Math.addExact(usedImageTokens, 2);
        float[] inputsEmbeds = new float[Math.multiplyExact(sequenceLength, hiddenSize)];
        System.arraycopy(textEmbeds.values(), 0, inputsEmbeds, 0, hiddenSize);
        System.arraycopy(
                vision.imageEmbeds(),
                0,
                inputsEmbeds,
                hiddenSize,
                Math.multiplyExact(usedImageTokens, hiddenSize));
        System.arraycopy(
                textEmbeds.values(),
                hiddenSize,
                inputsEmbeds,
                Math.multiplyExact(sequenceLength - 1, hiddenSize),
                hiddenSize);

        Path decoderGraph = vision.plan().graphs().decoder();
        if (decoderGraph == null || !Files.isRegularFile(decoderGraph)) {
            throw new IllegalArgumentException("decoder graph not found: " + decoderGraph);
        }

        OrtEnvironment env = OrtEnvironment.getEnvironment("gollek-paddleocr-vl");
        long loadStarted = System.nanoTime();
        try (OrtSession.SessionOptions options = new OrtSession.SessionOptions()) {
            options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
            options.setIntraOpNumThreads(Math.max(1, threads));
            options.setInterOpNumThreads(1);
            try (OrtSession session = env.createSession(decoderGraph.toAbsolutePath().toString(), options)) {
                Duration decoderLoadDuration = Duration.ofNanos(System.nanoTime() - loadStarted);
                List<IoInfo> decoderInputs = describeInfo(session.getInputInfo());
                List<IoInfo> decoderDeclaredOutputs = describeInfo(session.getOutputInfo());
                Map<String, OnnxTensorLike> feeds = new LinkedHashMap<>();
                List<OnnxTensor> closeables = new ArrayList<>();
                try {
                    OnnxTensor inputEmbeds = OnnxTensor.createTensor(
                            env,
                            FloatBuffer.wrap(inputsEmbeds),
                            new long[] { 1, sequenceLength, hiddenSize });
                    closeables.add(inputEmbeds);
                    feeds.put("inputs_embeds", inputEmbeds);

                    long[] attention = new long[sequenceLength];
                    java.util.Arrays.fill(attention, 1L);
                    OnnxTensor attentionMask = OnnxTensor.createTensor(
                            env,
                            LongBuffer.wrap(attention),
                            new long[] { 1, sequenceLength });
                    closeables.add(attentionMask);
                    feeds.put("attention_mask", attentionMask);

                    for (IoInfo input : decoderInputs) {
                        if (input.name() == null || !input.name().startsWith("past_key_values.")) {
                            continue;
                        }
                        long[] shape = emptyPastShape(input.shape());
                        OnnxTensor emptyPast = OnnxTensor.createTensor(
                                env,
                                FloatBuffer.wrap(new float[0]),
                                shape);
                        closeables.add(emptyPast);
                        feeds.put(input.name(), emptyPast);
                    }

                    long runStarted = System.nanoTime();
                    try (OrtSession.Result result = session.run(feeds)) {
                        List<IoInfo> decoderOutputs = describeResult(result);
                        DecoderLogitsSummary logitsSummary = summarizeLogits(result);
                        return new DecoderPrefillProbeResult(
                                vision.plan(),
                                vision.imageTensor(),
                                vision.graph(),
                                decoderGraph.toAbsolutePath().normalize(),
                                vision.loadDuration(),
                                vision.runDuration(),
                                textEmbeds.graph(),
                                textEmbeds.loadDuration(),
                                textEmbeds.runDuration(),
                                textEmbeds.shape(),
                                decoderLoadDuration,
                                Duration.ofNanos(System.nanoTime() - runStarted),
                                Math.toIntExact(imageTokenCount),
                                usedImageTokens,
                                sequenceLength,
                                hiddenSize,
                                decoderInputs,
                                decoderDeclaredOutputs,
                                decoderOutputs,
                                logitsSummary);
                    }
                } finally {
                    for (int i = closeables.size() - 1; i >= 0; i--) {
                        closeables.get(i).close();
                    }
                }
            }
        }
    }

    private static PromptDecodeProbeResult runPromptDecodeChecked(
            Path modelDir,
            Path imagePath,
            String userPrompt,
            String requestedVariant,
            int imageTokenLimit,
            int maxNewTokens,
            int threads) throws OrtException, IOException {
        VisionEmbedsResult vision = runVisionEncoderWithEmbeds(modelDir, imagePath, requestedVariant, threads);
        long imageTokenCount = vision.imageEmbedsShape().length == 0 ? 0 : vision.imageEmbedsShape()[0];
        if (imageTokenCount <= 0) {
            throw new IllegalArgumentException("vision_encoder produced no image embeddings");
        }
        int usedImageTokens = imageTokenLimit <= 0
                ? Math.toIntExact(imageTokenCount)
                : (int) Math.min(imageTokenCount, imageTokenLimit);

        Tokenizer tokenizer = TokenizerFactory.load(vision.plan().modelDir(), null);
        int imageTokenId = imageTokenId(tokenizer);
        String promptText = paddleOcrChatPrompt(userPrompt, usedImageTokens);
        long[] inputIds = tokenizer.encode(promptText, EncodeOptions.defaultOptions());
        int imageTokenPositions = countToken(inputIds, imageTokenId);
        if (imageTokenPositions != usedImageTokens) {
            throw new IllegalArgumentException("prompt tokenizer produced " + imageTokenPositions
                    + " image token(s), expected " + usedImageTokens
                    + ". Check tokenizer special token handling for <|IMAGE_PLACEHOLDER|>.");
        }

        GraphTensorResult promptEmbeds = runEmbedding(vision.plan().graphs().embedding(), threads, inputIds);
        int hiddenSize = hiddenSize(vision.imageEmbedsShape(), promptEmbeds.shape());
        int sequenceLength = inputIds.length;
        float[] inputsEmbeds = promptEmbeds.values().clone();
        replaceImageTokenEmbeddings(
                inputsEmbeds,
                inputIds,
                imageTokenId,
                vision.imageEmbeds(),
                usedImageTokens,
                hiddenSize);

        Path decoderGraph = vision.plan().graphs().decoder();
        if (decoderGraph == null || !Files.isRegularFile(decoderGraph)) {
            throw new IllegalArgumentException("decoder graph not found: " + decoderGraph);
        }

        OrtEnvironment env = OrtEnvironment.getEnvironment("gollek-paddleocr-vl");
        long loadStarted = System.nanoTime();
        try (OrtSession.SessionOptions options = new OrtSession.SessionOptions()) {
            options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
            options.setIntraOpNumThreads(Math.max(1, threads));
            options.setInterOpNumThreads(1);
            try (OrtSession session = env.createSession(decoderGraph.toAbsolutePath().toString(), options)) {
                Duration decoderLoadDuration = Duration.ofNanos(System.nanoTime() - loadStarted);
                List<IoInfo> decoderInputs = describeInfo(session.getInputInfo());
                List<IoInfo> decoderDeclaredOutputs = describeInfo(session.getOutputInfo());
                DecoderFeeds prefillFeeds = decoderFeeds(
                        env,
                        decoderInputs,
                        inputsEmbeds,
                        sequenceLength,
                        hiddenSize,
                        sequenceLength,
                        null);
                long prefillStarted = System.nanoTime();
                OrtSession.Result current = null;
                try {
                    current = session.run(prefillFeeds.feeds());
                } finally {
                    prefillFeeds.close();
                }
                Duration prefillDuration = Duration.ofNanos(System.nanoTime() - prefillStarted);
                List<IoInfo> decoderOutputs = describeResult(current);
                DecoderLogitsSummary logitsSummary = summarizeLogits(current);
                List<Integer> generated = new ArrayList<>();
                String finishReason = "length";
                long decodeStarted = System.nanoTime();
                try {
                    int decodeLimit = Math.max(0, maxNewTokens);
                    int nextToken = decodeLimit == 0 ? -1 : selectNextTokenId(current);
                    for (int step = 0; step < decodeLimit; step++) {
                        generated.add(nextToken);
                        if (isStopToken(tokenizer, nextToken)) {
                            finishReason = "stop";
                            break;
                        }
                        if (step + 1 >= decodeLimit) {
                            break;
                        }

                        GraphTensorResult nextEmbeds =
                                runEmbedding(vision.plan().graphs().embedding(), threads, new long[] { nextToken });
                        DecoderFeeds nextFeeds = decoderFeeds(
                                env,
                                decoderInputs,
                                nextEmbeds.values(),
                                1,
                                hiddenSize,
                                sequenceLength + generated.size(),
                                current);
                        OrtSession.Result previous = current;
                        try {
                            current = session.run(nextFeeds.feeds());
                        } finally {
                            nextFeeds.close();
                            previous.close();
                        }
                        nextToken = selectNextTokenId(current);
                    }
                } finally {
                    if (current != null) {
                        current.close();
                    }
                }

                long[] generatedIds = generated.stream().mapToLong(Integer::longValue).toArray();
                String decodedText = tokenizer.decode(
                        generatedIds,
                        DecodeOptions.builder().skipSpecialTokens(true).build());
                return new PromptDecodeProbeResult(
                        vision.plan(),
                        vision.imageTensor(),
                        vision.graph(),
                        decoderGraph.toAbsolutePath().normalize(),
                        vision.loadDuration(),
                        vision.runDuration(),
                        promptEmbeds.graph(),
                        promptEmbeds.loadDuration(),
                        promptEmbeds.runDuration(),
                        promptEmbeds.shape(),
                        decoderLoadDuration,
                        prefillDuration,
                        Duration.ofNanos(System.nanoTime() - decodeStarted),
                        Math.toIntExact(imageTokenCount),
                        usedImageTokens,
                        sequenceLength,
                        hiddenSize,
                        promptText,
                        inputIds,
                        imageTokenId,
                        imageTokenPositions,
                        generatedIds,
                        decodedText,
                        finishReason,
                        decoderInputs,
                        decoderDeclaredOutputs,
                        decoderOutputs,
                        logitsSummary);
            }
        }
    }

    private static VisionEmbedsResult runVisionEncoderWithEmbeds(
            Path modelDir,
            Path imagePath,
            String requestedVariant,
            int threads) throws OrtException {
        Objects.requireNonNull(modelDir, "modelDir");
        Objects.requireNonNull(imagePath, "imagePath");

        PaddleOcrVlOnnxPlanner.Plan plan =
                PaddleOcrVlOnnxPlanner.plan(modelDir, List.of(imagePath), requestedVariant);
        if (plan.images().isEmpty()) {
            throw new IllegalArgumentException("vision encoder probe needs one input image");
        }
        Path visionGraph = plan.graphs().visionEncoder();
        if (visionGraph == null || !Files.isRegularFile(visionGraph)) {
            throw new IllegalArgumentException("vision_encoder graph not found: " + visionGraph);
        }

        PaddleOcrVlOnnxPlanner.ImageTensor imageTensor =
                PaddleOcrVlOnnxPlanner.preprocessImage(imagePath, plan.processor());
        OrtEnvironment env = OrtEnvironment.getEnvironment("gollek-paddleocr-vl");

        long loadStarted = System.nanoTime();
        try (OrtSession.SessionOptions options = new OrtSession.SessionOptions()) {
            options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
            options.setIntraOpNumThreads(Math.max(1, threads));
            options.setInterOpNumThreads(1);
            try (OrtSession session = env.createSession(visionGraph.toAbsolutePath().toString(), options)) {
                Duration loadDuration = Duration.ofNanos(System.nanoTime() - loadStarted);
                List<IoInfo> inputs = describeInfo(session.getInputInfo());
                List<IoInfo> declaredOutputs = describeInfo(session.getOutputInfo());
                String pixelInputName = matchingInputName(inputs, "pixel_values");
                String gridInputName = matchingInputName(inputs, "image_grid_thw");
                if (pixelInputName == null || gridInputName == null) {
                    throw new IllegalArgumentException("vision_encoder inputs do not include pixel_values and image_grid_thw: "
                            + inputNames(inputs));
                }

                try (OnnxTensor pixelValues = OnnxTensor.createTensor(
                        env,
                        FloatBuffer.wrap(imageTensor.pixelValues()),
                        imageTensor.pixelValuesShape());
                     OnnxTensor imageGridThw = createGridTensor(
                             env,
                             imageTensor,
                             inputType(inputs, gridInputName))) {
                    Map<String, OnnxTensorLike> feeds = new LinkedHashMap<>();
                    feeds.put(pixelInputName, pixelValues);
                    feeds.put(gridInputName, imageGridThw);

                    long runStarted = System.nanoTime();
                    try (OrtSession.Result result = session.run(feeds)) {
                        Duration runDuration = Duration.ofNanos(System.nanoTime() - runStarted);
                        OnnxTensor imageEmbeds = tensor(result, "image_embeds");
                        TensorInfo info = imageEmbeds.getInfo();
                        return new VisionEmbedsResult(
                                plan,
                                imageTensor,
                                visionGraph.toAbsolutePath().normalize(),
                                loadDuration,
                                runDuration,
                                inputs,
                                declaredOutputs,
                                describeResult(result),
                                readFloatTensor(imageEmbeds),
                                info.getShape());
                    }
                }
            }
        }
    }

    private static GraphTensorResult runEmbedding(Path embeddingGraph, int threads, long[] inputIds)
            throws OrtException {
        if (embeddingGraph == null || !Files.isRegularFile(embeddingGraph)) {
            throw new IllegalArgumentException("embedding graph not found: " + embeddingGraph);
        }
        OrtEnvironment env = OrtEnvironment.getEnvironment("gollek-paddleocr-vl");
        long loadStarted = System.nanoTime();
        try (OrtSession.SessionOptions options = new OrtSession.SessionOptions()) {
            options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
            options.setIntraOpNumThreads(Math.max(1, threads));
            options.setInterOpNumThreads(1);
            try (OrtSession session = env.createSession(embeddingGraph.toAbsolutePath().toString(), options)) {
                Duration loadDuration = Duration.ofNanos(System.nanoTime() - loadStarted);
                String inputName = session.getInputNames().contains("input_ids")
                        ? "input_ids"
                        : session.getInputNames().iterator().next();
                try (OnnxTensor ids = OnnxTensor.createTensor(
                        env,
                        LongBuffer.wrap(inputIds),
                        new long[] { 1, inputIds.length })) {
                    long runStarted = System.nanoTime();
                    try (OrtSession.Result result = session.run(Map.of(inputName, ids))) {
                        OnnxTensor embeddings = tensor(result, "embeddings");
                        return new GraphTensorResult(
                                embeddingGraph.toAbsolutePath().normalize(),
                                loadDuration,
                                Duration.ofNanos(System.nanoTime() - runStarted),
                                readFloatTensor(embeddings),
                                embeddings.getInfo().getShape());
                    }
                }
            }
        }
    }

    private static int hiddenSize(long[] imageEmbedsShape, long[] textEmbedsShape) {
        long imageHidden = imageEmbedsShape.length == 0 ? -1 : imageEmbedsShape[imageEmbedsShape.length - 1];
        long textHidden = textEmbedsShape.length == 0 ? -1 : textEmbedsShape[textEmbedsShape.length - 1];
        if (imageHidden > 0 && textHidden > 0 && imageHidden != textHidden) {
            throw new IllegalArgumentException("vision hidden size " + imageHidden
                    + " does not match text embedding hidden size " + textHidden);
        }
        long hidden = imageHidden > 0 ? imageHidden : textHidden;
        if (hidden <= 0 || hidden > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("invalid decoder hidden size: " + hidden);
        }
        return (int) hidden;
    }

    private static long[] emptyPastShape(long[] declaredShape) {
        if (declaredShape == null || declaredShape.length == 0) {
            return new long[] { 1, 2, 0, 128 };
        }
        long[] shape = declaredShape.clone();
        for (int i = 0; i < shape.length; i++) {
            if (i == 0 && shape[i] <= 0) {
                shape[i] = 1;
            } else if (i == 1 && shape[i] <= 0) {
                shape[i] = 2;
            } else if (i == 2) {
                shape[i] = 0;
            } else if (i == 3 && shape[i] <= 0) {
                shape[i] = 128;
            } else if (shape[i] < 0) {
                shape[i] = 0;
            }
        }
        return shape;
    }

    private static OnnxTensor tensor(OrtSession.Result result, String preferredName) throws OrtException {
        OnnxValue preferred = result.get(preferredName).orElse(null);
        if (preferred instanceof OnnxTensor tensor) {
            return tensor;
        }
        for (Map.Entry<String, OnnxValue> entry : result) {
            if (entry.getValue() instanceof OnnxTensor tensor) {
                return tensor;
            }
        }
        throw new IllegalArgumentException("ONNX result did not contain a tensor output");
    }

    private static float[] readFloatTensor(OnnxTensor tensor) {
        FloatBuffer buffer = tensor.getFloatBuffer().duplicate();
        buffer.rewind();
        float[] values = new float[buffer.remaining()];
        buffer.get(values);
        return values;
    }

    private static String paddleOcrChatPrompt(String userPrompt, int imageTokens) {
        String text = userPrompt == null || userPrompt.isBlank()
                ? "Extract all text from the image."
                : userPrompt.trim();
        return "<|begin_of_sentence|>User: "
                + "<|IMAGE_START|>"
                + "<|IMAGE_PLACEHOLDER|>".repeat(Math.max(1, imageTokens))
                + "<|IMAGE_END|>"
                + text
                + "\nAssistant:\n";
    }

    private static int imageTokenId(Tokenizer tokenizer) {
        Integer id = tokenizer.specialTokens().get("<|IMAGE_PLACEHOLDER|>");
        return id == null ? 100_295 : id;
    }

    private static int countToken(long[] inputIds, int tokenId) {
        int count = 0;
        for (long inputId : inputIds) {
            if (inputId == tokenId) {
                count++;
            }
        }
        return count;
    }

    private static void replaceImageTokenEmbeddings(
            float[] inputsEmbeds,
            long[] inputIds,
            int imageTokenId,
            float[] imageEmbeds,
            int imageTokens,
            int hiddenSize) {
        int imageIndex = 0;
        for (int tokenIndex = 0; tokenIndex < inputIds.length; tokenIndex++) {
            if (inputIds[tokenIndex] != imageTokenId) {
                continue;
            }
            if (imageIndex >= imageTokens) {
                break;
            }
            System.arraycopy(
                    imageEmbeds,
                    Math.multiplyExact(imageIndex, hiddenSize),
                    inputsEmbeds,
                    Math.multiplyExact(tokenIndex, hiddenSize),
                    hiddenSize);
            imageIndex++;
        }
    }

    private static DecoderFeeds decoderFeeds(
            OrtEnvironment env,
            List<IoInfo> decoderInputs,
            float[] inputEmbeds,
            int inputTokenCount,
            int hiddenSize,
            int attentionLength,
            OrtSession.Result pastResult) throws OrtException {
        Map<String, OnnxTensorLike> feeds = new LinkedHashMap<>();
        List<OnnxTensor> closeables = new ArrayList<>();

        OnnxTensor inputEmbedsTensor = OnnxTensor.createTensor(
                env,
                FloatBuffer.wrap(inputEmbeds),
                new long[] { 1, inputTokenCount, hiddenSize });
        closeables.add(inputEmbedsTensor);
        feeds.put("inputs_embeds", inputEmbedsTensor);

        long[] attention = new long[attentionLength];
        Arrays.fill(attention, 1L);
        OnnxTensor attentionMask = OnnxTensor.createTensor(
                env,
                LongBuffer.wrap(attention),
                new long[] { 1, attentionLength });
        closeables.add(attentionMask);
        feeds.put("attention_mask", attentionMask);

        for (IoInfo input : decoderInputs) {
            if (input.name() == null || !input.name().startsWith("past_key_values.")) {
                continue;
            }
            if (pastResult == null) {
                long[] shape = emptyPastShape(input.shape());
                OnnxTensor emptyPast = OnnxTensor.createTensor(
                        env,
                        FloatBuffer.wrap(new float[0]),
                        shape);
                closeables.add(emptyPast);
                feeds.put(input.name(), emptyPast);
            } else {
                String outputName = "present." + input.name().substring("past_key_values.".length());
                OnnxValue value = pastResult.get(outputName).orElseThrow(
                        () -> new IllegalArgumentException("decoder output not found for " + input.name()
                                + " (expected " + outputName + ")"));
                if (!(value instanceof OnnxTensorLike tensor)) {
                    throw new IllegalArgumentException("decoder output " + outputName
                            + " is not a tensor-like value");
                }
                feeds.put(input.name(), tensor);
            }
        }
        return new DecoderFeeds(feeds, closeables);
    }

    private static int selectNextTokenId(OrtSession.Result result) throws OrtException {
        OnnxTensor logits = tensor(result, "logits");
        TensorInfo info = logits.getInfo();
        if (info.type != OnnxJavaType.FLOAT) {
            throw new IllegalArgumentException("decoder logits must be float, got " + info.type);
        }
        long[] shape = info.getShape();
        if (shape.length < 2 || shape[shape.length - 1] <= 0) {
            throw new IllegalArgumentException("invalid decoder logits shape: " + Arrays.toString(shape));
        }
        int vocabSize = Math.toIntExact(shape[shape.length - 1]);
        int sequenceIndex = shape.length >= 2 && shape[shape.length - 2] > 0
                ? Math.toIntExact(shape[shape.length - 2] - 1)
                : 0;
        FloatBuffer buffer = logits.getFloatBuffer().duplicate();
        long base = (long) sequenceIndex * vocabSize;
        if (base < 0 || base + vocabSize > buffer.capacity()) {
            throw new IllegalArgumentException("decoder logits buffer is smaller than declared shape: "
                    + Arrays.toString(shape));
        }
        buffer.position(Math.toIntExact(base));
        int bestToken = 0;
        float bestLogit = Float.NEGATIVE_INFINITY;
        for (int token = 0; token < vocabSize; token++) {
            float logit = buffer.get();
            if (Float.isFinite(logit) && logit > bestLogit) {
                bestLogit = logit;
                bestToken = token;
            }
        }
        return bestToken;
    }

    private static boolean isStopToken(Tokenizer tokenizer, int tokenId) {
        if (tokenId == tokenizer.eosTokenId()) {
            return true;
        }
        for (int stopTokenId : tokenizer.allStopTokenIds()) {
            if (tokenId == stopTokenId) {
                return true;
            }
        }
        return false;
    }

    private static List<Integer> extractLocationTokens(String text) {
        if (text == null || text.isEmpty()) {
            return List.of();
        }
        Matcher matcher = LOC_TOKEN_PATTERN.matcher(text);
        List<Integer> values = new ArrayList<>();
        while (matcher.find()) {
            try {
                values.add(Integer.parseInt(matcher.group(1)));
            } catch (NumberFormatException ignored) {
                // Ignore malformed location tokens in best-effort postprocessing.
            }
        }
        return List.copyOf(values);
    }

    private static List<LocationBox> locationBoxes(
            List<Integer> locations,
            PaddleOcrVlOnnxPlanner.ImagePlan image) {
        if (locations == null || locations.size() < 4) {
            return List.of();
        }
        int width = image == null ? 0 : image.originalWidth();
        int height = image == null ? 0 : image.originalHeight();
        List<LocationBox> boxes = new ArrayList<>();
        for (int i = 0; i + 3 < locations.size(); i += 4) {
            int x1 = locations.get(i);
            int y1 = locations.get(i + 1);
            int x2 = locations.get(i + 2);
            int y2 = locations.get(i + 3);
            boxes.add(new LocationBox(
                    boxes.size() + 1,
                    x1,
                    y1,
                    x2,
                    y2,
                    pixelCoordinate(x1, width),
                    pixelCoordinate(y1, height),
                    pixelCoordinate(x2, width),
                    pixelCoordinate(y2, height)));
        }
        return List.copyOf(boxes);
    }

    private static int pixelCoordinate(int loc, int span) {
        if (span <= 0) {
            return -1;
        }
        return Math.round(Math.max(0, Math.min(1000, loc)) * span / 1000.0f);
    }

    private static List<OcrTextRegion> textRegions(String raw, List<LocationBox> boxes) {
        if (raw == null || raw.isBlank() || boxes == null || boxes.isEmpty()) {
            return List.of();
        }
        Matcher matcher = LOC_TOKEN_PATTERN.matcher(raw);
        List<OcrTextRegion> regions = new ArrayList<>();
        StringBuilder pendingText = new StringBuilder();
        int cursor = 0;
        int locCount = 0;
        while (matcher.find()) {
            if (matcher.start() > cursor) {
                pendingText.append(raw, cursor, matcher.start());
            }
            cursor = matcher.end();
            locCount++;
            if (locCount % 4 == 0) {
                int boxIndex = (locCount / 4) - 1;
                if (boxIndex < boxes.size()) {
                    regions.add(new OcrTextRegion(
                            regions.size() + 1,
                            cleanOcrRegionText(pendingText.toString()),
                            boxes.get(boxIndex)));
                    pendingText.setLength(0);
                }
            }
        }
        if (cursor < raw.length()) {
            pendingText.append(raw.substring(cursor));
        }
        String trailingText = cleanOcrRegionText(pendingText.toString());
        if (!trailingText.isBlank() && !regions.isEmpty()) {
            OcrTextRegion last = regions.get(regions.size() - 1);
            String mergedText = last.text().isBlank()
                    ? trailingText
                    : last.text() + " " + trailingText;
            regions.set(regions.size() - 1, new OcrTextRegion(last.index(), mergedText, last.box()));
        }
        return List.copyOf(regions);
    }

    private static String cleanOcrRegionText(String text) {
        if (text == null || text.isBlank()) {
            return "";
        }
        return text
                .replaceAll("<\\|[^>]+\\|>", " ")
                .replaceAll("\\s+", " ")
                .trim();
    }

    private static String ocrDisplayText(
            String raw,
            String textWithoutLocations,
            List<Integer> locations,
            List<LocationBox> boxes,
            List<OcrTextRegion> regions) {
        if (locations == null || locations.isEmpty()) {
            return raw == null ? "" : raw.trim();
        }
        StringBuilder out = new StringBuilder();
        boolean hasRegionText = regions != null && regions.stream().anyMatch(region -> !region.text().isBlank());
        if (hasRegionText) {
            out.append("OCR regions:");
            for (OcrTextRegion region : regions) {
                out.append(System.lineSeparator())
                        .append("- #").append(region.index());
                if (!region.text().isBlank()) {
                    out.append(" \"").append(region.text()).append("\"");
                }
                appendBoxSummary(out, region.box());
            }
        } else if (textWithoutLocations != null && !textWithoutLocations.isBlank()) {
            out.append(textWithoutLocations.trim()).append(System.lineSeparator()).append(System.lineSeparator());
        }
        if (!hasRegionText && !boxes.isEmpty()) {
            out.append("Detected location boxes:");
            for (LocationBox box : boxes) {
                out.append(System.lineSeparator())
                        .append("- #").append(box.index());
                appendBoxSummary(out, box);
            }
        }
        int remainder = locations.size() % 4;
        if (remainder != 0 || boxes.isEmpty()) {
            if (out.length() > 0) {
                out.append(System.lineSeparator());
            }
            out.append("Location tokens: ").append(locations);
        }
        return out.toString();
    }

    private static void appendBoxSummary(StringBuilder out, LocationBox box) {
        out.append(" loc=[")
                .append(box.x1()).append(",")
                .append(box.y1()).append(",")
                .append(box.x2()).append(",")
                .append(box.y2()).append("]")
                .append(" box=[")
                .append(box.normalizedX1()).append(",")
                .append(box.normalizedY1()).append(",")
                .append(box.normalizedX2()).append(",")
                .append(box.normalizedY2()).append("]");
        if (box.pixelX1() >= 0) {
            out.append(" pixels=[")
                    .append(box.normalizedPixelX1()).append(",")
                    .append(box.normalizedPixelY1()).append(",")
                    .append(box.normalizedPixelX2()).append(",")
                    .append(box.normalizedPixelY2()).append("]");
        }
    }

    private static OnnxTensor createGridTensor(
            OrtEnvironment env,
            PaddleOcrVlOnnxPlanner.ImageTensor imageTensor,
            OnnxJavaType type) throws OrtException {
        if (type == OnnxJavaType.INT32) {
            int[] values = new int[imageTensor.imageGridThw().length];
            for (int i = 0; i < values.length; i++) {
                values[i] = Math.toIntExact(imageTensor.imageGridThw()[i]);
            }
            return OnnxTensor.createTensor(env, IntBuffer.wrap(values), imageTensor.imageGridThwShape());
        }
        return OnnxTensor.createTensor(env, LongBuffer.wrap(imageTensor.imageGridThw()), imageTensor.imageGridThwShape());
    }

    private static List<IoInfo> describeInfo(Map<String, NodeInfo> info) {
        List<IoInfo> values = new ArrayList<>();
        for (Map.Entry<String, NodeInfo> entry : info.entrySet()) {
            ValueInfo valueInfo = entry.getValue().getInfo();
            values.add(describeValue(entry.getKey(), valueInfo));
        }
        return List.copyOf(values);
    }

    private static List<IoInfo> describeResult(OrtSession.Result result) throws OrtException {
        List<IoInfo> values = new ArrayList<>();
        for (Map.Entry<String, OnnxValue> entry : result) {
            OnnxValue value = entry.getValue();
            if (value instanceof OnnxTensor tensor) {
                TensorInfo info = tensor.getInfo();
                values.add(new IoInfo(
                        entry.getKey(),
                        "tensor",
                        info.type == null ? "unknown" : info.type.name().toLowerCase(),
                        info.getShape(),
                        info.getNumElements()));
            } else {
                values.add(new IoInfo(entry.getKey(), String.valueOf(value.getType()), "unknown", new long[0], -1));
            }
        }
        return List.copyOf(values);
    }

    private static DecoderLogitsSummary summarizeLogits(OrtSession.Result result) throws OrtException {
        String name = null;
        OnnxTensor logits = null;
        OnnxValue preferred = result.get("logits").orElse(null);
        if (preferred instanceof OnnxTensor tensor) {
            name = "logits";
            logits = tensor;
        }
        if (logits == null) {
            for (Map.Entry<String, OnnxValue> entry : result) {
                if (entry.getKey() != null
                        && entry.getKey().toLowerCase(java.util.Locale.ROOT).contains("logit")
                        && entry.getValue() instanceof OnnxTensor tensor) {
                    name = entry.getKey();
                    logits = tensor;
                    break;
                }
            }
        }
        if (logits == null) {
            return null;
        }

        TensorInfo info = logits.getInfo();
        long[] shape = info.getShape();
        int vocabSize = shape.length == 0 || shape[shape.length - 1] <= 0
                ? -1
                : Math.toIntExact(shape[shape.length - 1]);
        int sequenceIndex = shape.length >= 2 && shape[shape.length - 2] > 0
                ? Math.toIntExact(shape[shape.length - 2] - 1)
                : 0;
        int[] topTokenIds = new int[0];
        float[] topLogits = new float[0];
        if (info.type == OnnxJavaType.FLOAT && vocabSize > 0) {
            topTokenIds = new int[5];
            topLogits = new float[5];
            java.util.Arrays.fill(topTokenIds, -1);
            java.util.Arrays.fill(topLogits, Float.NEGATIVE_INFINITY);
            FloatBuffer buffer = logits.getFloatBuffer().duplicate();
            long base = (long) sequenceIndex * vocabSize;
            if (base <= Integer.MAX_VALUE && base + vocabSize <= buffer.capacity()) {
                buffer.position(Math.toIntExact(base));
                for (int tokenId = 0; tokenId < vocabSize; tokenId++) {
                    offerTopLogit(tokenId, buffer.get(), topTokenIds, topLogits);
                }
            }
        }
        return new DecoderLogitsSummary(
                name,
                info.type == null ? "unknown" : info.type.name().toLowerCase(java.util.Locale.ROOT),
                shape,
                sequenceIndex,
                vocabSize,
                topTokenIds,
                topLogits);
    }

    private static void offerTopLogit(int tokenId, float logit, int[] tokenIds, float[] logits) {
        if (!Float.isFinite(logit) || tokenIds.length == 0 || logit <= logits[logits.length - 1]) {
            return;
        }
        int insert = logits.length - 1;
        while (insert > 0 && logit > logits[insert - 1]) {
            logits[insert] = logits[insert - 1];
            tokenIds[insert] = tokenIds[insert - 1];
            insert--;
        }
        logits[insert] = logit;
        tokenIds[insert] = tokenId;
    }

    private static IoInfo describeValue(String name, ValueInfo valueInfo) {
        if (valueInfo instanceof TensorInfo tensorInfo) {
            return new IoInfo(
                    name,
                    "tensor",
                    tensorInfo.type == null ? "unknown" : tensorInfo.type.name().toLowerCase(),
                    tensorInfo.getShape(),
                    tensorInfo.getNumElements());
        }
        return new IoInfo(name, valueInfo == null ? "unknown" : valueInfo.getClass().getSimpleName(), "unknown",
                new long[0], -1);
    }

    private static String matchingInputName(List<IoInfo> inputs, String expected) {
        for (IoInfo input : inputs) {
            if (expected.equals(input.name())) {
                return input.name();
            }
        }
        for (IoInfo input : inputs) {
            if (input.name() != null && input.name().toLowerCase().contains(expected)) {
                return input.name();
            }
        }
        return null;
    }

    private static OnnxJavaType inputType(List<IoInfo> inputs, String name) {
        for (IoInfo input : inputs) {
            if (Objects.equals(input.name(), name)) {
                return switch (input.type()) {
                    case "int32" -> OnnxJavaType.INT32;
                    case "int64" -> OnnxJavaType.INT64;
                    default -> OnnxJavaType.UNKNOWN;
                };
            }
        }
        return OnnxJavaType.UNKNOWN;
    }

    private static String inputNames(List<IoInfo> inputs) {
        List<String> names = new ArrayList<>();
        for (IoInfo input : inputs) {
            names.add(input.name());
        }
        return String.join(", ", names);
    }

    public record VisionEncoderProbeResult(
            PaddleOcrVlOnnxPlanner.Plan plan,
            PaddleOcrVlOnnxPlanner.ImageTensor imageTensor,
            Path graph,
            Duration loadDuration,
            Duration runDuration,
            List<IoInfo> inputs,
            List<IoInfo> declaredOutputs,
            List<IoInfo> outputs) {
    }

    private record VisionEmbedsResult(
            PaddleOcrVlOnnxPlanner.Plan plan,
            PaddleOcrVlOnnxPlanner.ImageTensor imageTensor,
            Path graph,
            Duration loadDuration,
            Duration runDuration,
            List<IoInfo> inputs,
            List<IoInfo> declaredOutputs,
            List<IoInfo> outputs,
            float[] imageEmbeds,
            long[] imageEmbedsShape) {
    }

    public record GraphIoProbeResult(
            Path graph,
            Duration loadDuration,
            List<IoInfo> inputs,
            List<IoInfo> outputs) {
    }

    private record GraphTensorResult(
            Path graph,
            Duration loadDuration,
            Duration runDuration,
            float[] values,
            long[] shape) {
    }

    private record DecoderFeeds(
            Map<String, OnnxTensorLike> feeds,
            List<OnnxTensor> closeables) implements AutoCloseable {
        @Override
        public void close() throws OrtException {
            for (int i = closeables.size() - 1; i >= 0; i--) {
                closeables.get(i).close();
            }
        }
    }

    public record DecoderPrefillProbeResult(
            PaddleOcrVlOnnxPlanner.Plan plan,
            PaddleOcrVlOnnxPlanner.ImageTensor imageTensor,
            Path visionGraph,
            Path decoderGraph,
            Duration visionLoadDuration,
            Duration visionRunDuration,
            Path embeddingGraph,
            Duration embeddingLoadDuration,
            Duration embeddingRunDuration,
            long[] textEmbedsShape,
            Duration decoderLoadDuration,
            Duration decoderRunDuration,
            int imageTokens,
            int usedImageTokens,
            int sequenceLength,
            int hiddenSize,
            List<IoInfo> decoderInputs,
            List<IoInfo> decoderDeclaredOutputs,
            List<IoInfo> decoderOutputs,
            DecoderLogitsSummary decoderLogits) {
    }

    public record PromptDecodeProbeResult(
            PaddleOcrVlOnnxPlanner.Plan plan,
            PaddleOcrVlOnnxPlanner.ImageTensor imageTensor,
            Path visionGraph,
            Path decoderGraph,
            Duration visionLoadDuration,
            Duration visionRunDuration,
            Path embeddingGraph,
            Duration embeddingLoadDuration,
            Duration embeddingRunDuration,
            long[] promptEmbedsShape,
            Duration decoderLoadDuration,
            Duration decoderPrefillDuration,
            Duration decoderDecodeDuration,
            int imageTokens,
            int usedImageTokens,
            int sequenceLength,
            int hiddenSize,
            String promptText,
            long[] inputIds,
            int imageTokenId,
            int imageTokenPositions,
            long[] generatedTokenIds,
            String decodedText,
            String finishReason,
            List<IoInfo> decoderInputs,
            List<IoInfo> decoderDeclaredOutputs,
            List<IoInfo> decoderOutputs,
            DecoderLogitsSummary decoderLogits) {
    }

    public record OcrPostProcessResult(
            String rawText,
            String displayText,
            String textWithoutLocations,
            List<Integer> locations,
            List<LocationBox> boxes,
            List<OcrTextRegion> regions) {
    }

    public record OcrTextRegion(
            int index,
            String text,
            LocationBox box) {
    }

    public record LocationBox(
            int index,
            int x1,
            int y1,
            int x2,
            int y2,
            int pixelX1,
            int pixelY1,
            int pixelX2,
            int pixelY2) {
        public int normalizedX1() {
            return Math.min(x1, x2);
        }

        public int normalizedY1() {
            return Math.min(y1, y2);
        }

        public int normalizedX2() {
            return Math.max(x1, x2);
        }

        public int normalizedY2() {
            return Math.max(y1, y2);
        }

        public int normalizedPixelX1() {
            return normalizedPixelMin(pixelX1, pixelX2);
        }

        public int normalizedPixelY1() {
            return normalizedPixelMin(pixelY1, pixelY2);
        }

        public int normalizedPixelX2() {
            return normalizedPixelMax(pixelX1, pixelX2);
        }

        public int normalizedPixelY2() {
            return normalizedPixelMax(pixelY1, pixelY2);
        }

        private static int normalizedPixelMin(int first, int second) {
            if (first < 0 || second < 0) {
                return -1;
            }
            return Math.min(first, second);
        }

        private static int normalizedPixelMax(int first, int second) {
            if (first < 0 || second < 0) {
                return -1;
            }
            return Math.max(first, second);
        }
    }

    public record DecoderLogitsSummary(
            String name,
            String type,
            long[] shape,
            int sequenceIndex,
            int vocabSize,
            int[] topTokenIds,
            float[] topLogits) {
    }

    public record IoInfo(
            String name,
            String kind,
            String type,
            long[] shape,
            long elements) {
    }
}
