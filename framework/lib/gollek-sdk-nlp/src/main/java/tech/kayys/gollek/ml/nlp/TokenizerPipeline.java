package tech.kayys.gollek.ml.nlp;

import tech.kayys.gollek.tokenizer.TokenizerFactory;
import tech.kayys.gollek.tokenizer.spi.*;

import java.util.Arrays;

/**
 * Tokenization pipeline that bridges the SDK NLP layer to the Gollek tokenizer core.
 *
 * <p>Provides encode/decode operations using the model's tokenizer, loading it
 * from the model's directory via {@link TokenizerFactory}.
 *
 * <h3>Example</h3>
 * <pre>{@code
 * var tokenizer = new TokenizerPipeline("Qwen/Qwen2.5-0.5B", "/path/to/tokenizer.json");
 * long[] tokens = tokenizer.encode("Hello, world!");
 * String decoded = tokenizer.decode(tokens);
 * }</pre>
 *
 * @see TokenizerFactory
 * @see PipelineFactory
 */
public class TokenizerPipeline implements Pipeline<String, long[]> {

    private final String modelId;
    private final Tokenizer tokenizer;

    /**
     * Creates a tokenizer pipeline from a tokenizer path.
     *
     * @param modelId       model identifier for display/registry purposes
     * @param tokenizerPath absolute path to the tokenizer file (e.g. tokenizer.json)
     */
    public TokenizerPipeline(String modelId, String tokenizerPath) {
        this.modelId = modelId;
        ModelConfig config = new ModelConfig(TokenizerType.BPE, java.nio.file.Path.of(tokenizerPath));
        this.tokenizer = TokenizerFactory.create(config);
    }

    /**
     * Creates a tokenizer pipeline with an externally provided tokenizer.
     *
     * @param modelId   model identifier
     * @param tokenizer a pre-configured tokenizer instance
     */
    public TokenizerPipeline(String modelId, Tokenizer tokenizer) {
        this.modelId = modelId;
        this.tokenizer = tokenizer;
    }

    /**
     * Encodes text into token IDs.
     *
     * @param text the input text
     * @return array of token IDs
     */
    public long[] encode(String text) {
        return tokenizer.encode(text, new EncodeOptions());
    }

    /**
     * Encodes text with custom options (e.g. add special tokens, max length).
     *
     * @param text    the input text
     * @param options encoding options
     * @return array of token IDs
     */
    public long[] encode(String text, EncodeOptions options) {
        return tokenizer.encode(text, options);
    }

    /**
     * Decodes token IDs back to text.
     *
     * @param tokens array of token IDs
     * @return decoded text
     */
    public String decode(long[] tokens) {
        return tokenizer.decode(tokens, new DecodeOptions());
    }

    /**
     * Decodes token IDs with custom options.
     *
     * @param tokens  array of token IDs
     * @param options decoding options
     * @return decoded text
     */
    public String decode(long[] tokens, DecodeOptions options) {
        return tokenizer.decode(tokens, options);
    }

    /**
     * Returns the vocabulary size of the tokenizer.
     */
    public int vocabSize() {
        return tokenizer.vocabSize();
    }

    /**
     * Returns the beginning-of-sequence token ID.
     */
    public int bosTokenId() {
        return tokenizer.bosTokenId();
    }

    /**
     * Returns the end-of-sequence token ID.
     */
    public int eosTokenId() {
        return tokenizer.eosTokenId();
    }

    @Override
    public long[] process(String input) {
        return encode(input);
    }

    @Override
    public String task() {
        return "tokenization";
    }

    @Override
    public String model() {
        return modelId;
    }
}
