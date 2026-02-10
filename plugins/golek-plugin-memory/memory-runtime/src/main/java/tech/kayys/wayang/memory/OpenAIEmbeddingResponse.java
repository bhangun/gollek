package tech.kayys.gamelan.executor.memory;

import java.util.List;

/**
 * OpenAI embedding response
 */
public class OpenAIEmbeddingResponse {
    public String object;
    public List<EmbeddingData> data;
    public String model;
    public Usage usage;

    public static class EmbeddingData {
        public String object;
        public float[] embedding;
        public int index;
    }

    public static class Usage {
        public int prompt_tokens;
        public int total_tokens;
    }
}