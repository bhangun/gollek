package tech.kayys.gollek.ml.nlp;

/**
 * Base interface for all NLP pipelines.
 *
 * <p>Pipelines provide a high-level API for common NLP tasks, abstracting model
 * loading, tokenization, and post-processing behind a single {@link #process} call.
 *
 * <p>Implementations are created via {@link PipelineFactory#create(String, String)}
 * and should be closed after use to release native or network resources.
 *
 * <pre>{@code
 * try (var pipeline = PipelineFactory.<String, String>create("text-generation", "llama3")) {
 *     String result = pipeline.process("Explain transformers");
 * }
 * }</pre>
 *
 * @param <I> the input type (e.g. {@code String} for text tasks)
 * @param <O> the output type (e.g. {@code String}, {@code float[]}, or a result record)
 * @see PipelineFactory
 */
public interface Pipeline<I, O> {

    /**
     * Processes a single input through the pipeline.
     *
     * @param input the input to process; must not be {@code null}
     * @return the pipeline output
     * @throws PipelineException if processing fails
     */
    O process(I input);

    /**
     * Returns the task identifier this pipeline performs.
     *
     * @return task name, e.g. {@code "text-generation"}, {@code "embedding"}
     */
    String task();

    /**
     * Returns the model identifier used by this pipeline.
     *
     * @return model ID as passed to the factory or constructor
     */
    String model();

    /**
     * Releases any resources held by this pipeline (e.g. native memory, HTTP connections).
     * The default implementation is a no-op.
     */
    default void close() {}
}
