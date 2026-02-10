package tech.kayys.gamelan.executor.memory;

import java.util.List;

/**
 * OpenAI embedding request (batch)
 */
public class OpenAIEmbeddingBatchRequest {
    public final String model;
    public final List<String> input;
    public final String encoding_format;

    public OpenAIEmbeddingBatchRequest(String model, List<String> input, String encoding_format) {
        this.model = model;
        this.input = input;
        this.encoding_format = encoding_format;
    }
}