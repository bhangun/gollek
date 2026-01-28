package tech.kayys.golek.provider.openai;

import java.util.List;

/**
 * OpenAI embedding data
 */
public class OpenAIEmbeddingData {

    private String object;
    private List<Double> embedding;

    private intindex;

    public String getObject() { return object; }

    public void setObject(String object) {
        this.object = object;
    }

    public List<Double> getEmbedding() {
        return embedding;
    }

    public void setEmbedding(List<Double> embedding) {
        this.embedding = embedding;
    }

    public int getIndex() {
        return index;
    }

    public void setIndex(int index) {
        this.index = index;
    }
}
