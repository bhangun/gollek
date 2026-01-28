package tech.kayys.golek.provider.openai;

import java.util.List;

/**
 * OpenAI embedding response
 */
public class OpenAIEmbeddingResponse {

    private String object;
    private List<OpenAIEmbeddingData> data;
    private String model;
    private OpenAIUsage usage;

    public String getObject() {
        return object;
    }

    public void setObject(String object) {
        this.object = object;
    }

    public List<OpenAIEmbeddingData> getData() {
        return data;
    }

    public void setData(List<OpenAIEmbeddingData> data) {
        this.data = data;
    }

    public String getModel() {
        return model;
    }

    public void setModel(String model) {
        this.model = model;
    }

    public OpenAIUsage getUsage() {
        return usage;
    }

    public void setUsage(OpenAIUsage usage) {
        this.usage = usage;
    }
}
