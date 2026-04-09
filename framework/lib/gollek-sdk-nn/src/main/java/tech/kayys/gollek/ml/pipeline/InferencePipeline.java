package tech.kayys.gollek.ml.pipeline;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.nn.NNModule;
// import tech.kayys.gollek.ml.nlp.Tokenizer;  // TODO: nlp package not yet implemented

import java.util.List;
import java.util.function.Function;

/**
 * End-to-end inference pipeline — chains preprocessing, model inference,
 * and postprocessing into a single callable unit.
 *
 * <p>Equivalent to HuggingFace's {@code pipeline()} function.
 * Supports text classification, text generation, embeddings, and image classification.
 *
 * <h3>Example — text classification</h3>
 * <pre>{@code
 * var pipeline = InferencePipeline.<String, float[]>builder()
 *     .preprocess(text -> tokenizer.batchEncode(List.of(text), 128, true, true))
 *     .model(classifier)
 *     .postprocess(logits -> softmax(logits.data()))
 *     .build();
 *
 * float[] probs = pipeline.run("This movie was great!");
 * }</pre>
 *
 * <h3>Example — text generation</h3>
 * <pre>{@code
 * var pipeline = TextGenerationPipeline.builder()
 *     .tokenizer(tokenizer)
 *     .model(gpt2)
 *     .maxNewTokens(100)
 *     .temperature(0.8f)
 *     .build();
 *
 * String output = pipeline.generate("Once upon a time");
 * }</pre>
 *
 * @param <I> input type (e.g. {@code String}, {@code float[]})
 * @param <O> output type (e.g. {@code float[]}, {@code String})
 */
public final class InferencePipeline<I, O> {

    private final Function<I, GradTensor>  preprocess;
    private final NNModule                   model;
    private final Function<GradTensor, O>  postprocess;

    private InferencePipeline(Builder<I, O> b) {
        this.preprocess  = b.preprocess;
        this.model       = b.model;
        this.postprocess = b.postprocess;
    }

    /**
     * Runs the full pipeline: preprocess → model → postprocess.
     *
     * @param input raw input
     * @return processed output
     */
    public O run(I input) {
        model.eval();
        GradTensor tensor = preprocess.apply(input);
        GradTensor output = model.forward(tensor);
        return postprocess.apply(output);
    }

    /**
     * Runs the pipeline on a batch of inputs in parallel using virtual threads.
     *
     * @param inputs list of raw inputs
     * @return list of outputs in the same order
     */
    public List<O> runBatch(List<I> inputs) {
        return inputs.parallelStream().map(this::run).toList();
    }

    /** @return a new builder */
    public static <I, O> Builder<I, O> builder() { return new Builder<>(); }

    /**
     * Builder for {@link InferencePipeline}.
     *
     * @param <I> input type
     * @param <O> output type
     */
    public static final class Builder<I, O> {
        private Function<I, GradTensor>  preprocess;
        private NNModule                   model;
        private Function<GradTensor, O>  postprocess;

        /**
         * Sets the preprocessing function that converts raw input to a tensor.
         *
         * @param fn preprocessing function
         */
        public Builder<I, O> preprocess(Function<I, GradTensor> fn) { this.preprocess = fn; return this; }

        /**
         * Sets the model to run inference with.
         *
         * @param m trained model
         */
        public Builder<I, O> model(NNModule m) { this.model = m; return this; }

        /**
         * Sets the postprocessing function that converts model output to the desired type.
         *
         * @param fn postprocessing function
         */
        public Builder<I, O> postprocess(Function<GradTensor, O> fn) { this.postprocess = fn; return this; }

        /**
         * Builds the pipeline.
         *
         * @return configured {@link InferencePipeline}
         */
        public InferencePipeline<I, O> build() { return new InferencePipeline<>(this); }
    }

    // ── Convenience factory methods ───────────────────────────────────────

    /**
     * Creates a text classification pipeline.
     *
     * @param tokenizer  tokenizer for text encoding
     * @param model      classification model
     * @param maxLength  maximum token sequence length
     * @param numClasses number of output classes
     * @return pipeline that maps {@code String → float[numClasses]} probabilities
     */
    // TODO: Tokenizer from nlp package not yet implemented
    /*
    public static InferencePipeline<String, float[]> textClassification(
             Tokenizer tokenizer, NNModule model, int maxLength, int numClasses) {
         return InferencePipeline.<String, float[]>builder()
             .preprocess(text -> {
                 Tokenizer.BatchEncoding enc = tokenizer.batchEncode(
                     List.of(text), maxLength, true, true);
                 float[] ids = new float[maxLength];
                 for (int i = 0; i < maxLength; i++) ids[i] = enc.inputIds()[0][i];
                 return GradTensor.of(ids, 1, maxLength);
             })
             .model(model)
             .postprocess(logits -> {
                 float[] d = logits.data();
                 // Softmax
                 float max = Float.NEGATIVE_INFINITY;
                 for (float v : d) if (v > max) max = v;
                 float sum = 0; float[] probs = new float[d.length];
                 for (int i = 0; i < d.length; i++) { probs[i] = (float)Math.exp(d[i]-max); sum += probs[i]; }
                 for (int i = 0; i < d.length; i++) probs[i] /= sum;
                 return probs;
             })
             .build();
     }
    */

     /**
      * Creates an embedding pipeline.
      *
      * @param tokenizer tokenizer for text encoding
      * @param model     embedding model
      * @param maxLength maximum token sequence length
      * @return pipeline that maps {@code String → float[]} embedding vector
      */
    // TODO: Tokenizer from nlp package not yet implemented
    /*
    public static InferencePipeline<String, float[]> embedding(
             Tokenizer tokenizer, NNModule model, int maxLength) {
         return InferencePipeline.<String, float[]>builder()
             .preprocess(text -> {
                 Tokenizer.BatchEncoding enc = tokenizer.batchEncode(
                     List.of(text), maxLength, true, true);
                 float[] ids = new float[maxLength];
                 for (int i = 0; i < maxLength; i++) ids[i] = enc.inputIds()[0][i];
                 return GradTensor.of(ids, 1, maxLength);
             })
             .model(model)
             .postprocess(GradTensor::data)
             .build();
     }
    */
}
