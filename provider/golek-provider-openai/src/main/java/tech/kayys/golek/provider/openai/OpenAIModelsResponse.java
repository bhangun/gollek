package tech.kayys.golek.provider.openai;

import java.util.List;

/**
 * OpenAI models response
 */
public class OpenAIModelsResponse {

    private String object;
    private List<OpenAIModelData> data;

    public String getObject() {
        return object;
    }

    public void setObject(String object) {
        this.object = object;
    }

    public List<OpenAIModelData> getData() {
        return data;
    }

    public void setData(List<OpenAIModelData> data) {
        this.data = data;
    }
}
