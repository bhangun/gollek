package tech.kayys.gamelan.executor.memory;

import java.util.List;

/**
 * OpenAI embedding request (single)
 */
public class OpenAIEmbeddingRequest {
    public final String model;
    public final String input;
    public final String encoding_format;

    public OpenAIEmbeddingRequest(String model, String input, String encoding_format) {
        this.model = model;
        this.input = input;
        this.encoding_format = encoding_format;
    }
}