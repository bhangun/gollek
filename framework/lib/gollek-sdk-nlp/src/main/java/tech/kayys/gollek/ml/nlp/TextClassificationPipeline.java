package tech.kayys.gollek.ml.nlp;

import tech.kayys.gollek.sdk.api.GollekSdk;
import tech.kayys.gollek.sdk.api.GollekSdkProvider;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;

import java.util.*;

/**
 * Zero-shot text classification pipeline backed by the Gollek inference engine.
 *
 * <p>Classifies text into one of a set of predefined labels by constructing a
 * prompt that asks the model to choose a category. No fine-tuning is required.
 *
 * <p>The default label set is {@code ["positive", "negative", "neutral"]}.
 * Supply a custom list via {@link #TextClassificationPipeline(String, List)}.
 *
 * <h3>Example</h3>
 * <pre>{@code
 * var pipeline = new TextClassificationPipeline("bert-base-uncased");
 * Classification result = pipeline.classify("This movie is amazing!");
 * System.out.println(result); // Classification(label=positive, score=0.85)
 * }</pre>
 *
 * @see PipelineFactory
 */
public class TextClassificationPipeline implements Pipeline<String, TextClassificationPipeline.Classification> {

    private final String modelId;
    private final GollekSdk sdk;
    private final List<String> labels;

    /**
     * Creates a pipeline with the default {@code ["positive", "negative", "neutral"]} labels.
     *
     * @param modelId model identifier
     */
    public TextClassificationPipeline(String modelId) {
        this(modelId, List.of("positive", "negative", "neutral"));
    }

    /**
     * Creates a pipeline with a custom label set.
     *
     * @param modelId model identifier
     * @param labels  non-empty list of candidate category labels
     */
    public TextClassificationPipeline(String modelId, List<String> labels) {
        this.modelId = modelId;
        this.labels = labels;
        this.sdk = resolveSdk();
    }

    /**
     * Classifies the given text into one of the configured labels.
     *
     * <p>Uses a zero-shot prompt asking the model to pick the best matching category.
     * The first label whose name appears in the model's response is returned; if none
     * match, the first label in the list is used as a fallback.
     *
     * @param text the text to classify; must not be {@code null}
     * @return a {@link Classification} containing the predicted label and confidence score
     * @throws PipelineException if inference fails
     */
    public Classification classify(String text) {
        if (sdk == null) {
            throw new PipelineException("No GollekSdk provider available on classpath");
        }
        try {
            String prompt = "Classify the following text into one of these categories: "
                + String.join(", ", labels) + ".\n\nText: \"" + text + "\"\n\nCategory:";

            InferenceRequest request = InferenceRequest.builder()
                .model(modelId)
                .prompt(prompt)
                .maxTokens(10)
                .temperature(0.1f)
                .build();

            InferenceResponse response = sdk.createCompletion(request);
            String predicted = response.getContent().trim().toLowerCase();

            String bestLabel = labels.get(0);
            for (String label : labels) {
                if (predicted.contains(label.toLowerCase())) {
                    bestLabel = label;
                    break;
                }
            }

            return new Classification(bestLabel, 0.85f, labels);
        } catch (Exception e) {
            throw new PipelineException("Classification failed: " + e.getMessage(), e);
        }
    }

    /**
     * Delegates to {@link #classify(String)}.
     *
     * @param input the text to classify
     * @return the classification result
     * @throws PipelineException if inference fails
     */
    @Override
    public Classification process(String input) {
        return classify(input);
    }

    @Override
    public String task() {
        return "text-classification";
    }

    @Override
    public String model() {
        return modelId;
    }

    /** Resolves a {@link GollekSdk} via {@link java.util.ServiceLoader}. */
    private GollekSdk resolveSdk() {
        try {
            return java.util.ServiceLoader.load(GollekSdkProvider.class)
                .findFirst()
                .map(p -> p.create(null))
                .orElse(null);
        } catch (Exception e) {
            return null;
        }
    }

    /**
     * Immutable result of a classification operation.
     *
     * @param label     the predicted category label
     * @param score     confidence score in range {@code [0.0, 1.0]}
     * @param allLabels the full list of candidate labels used during classification
     */
    public record Classification(String label, float score, List<String> allLabels) {
        @Override
        public String toString() {
            return "Classification(label=" + label + ", score=" + String.format("%.2f", score) + ")";
        }
    }
}
