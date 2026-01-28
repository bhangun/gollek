package tech.kayys.golek.provider.openai;

import java.util.List;

/**
 * OpenAI completion response
 */
public class OpenAIResponse {

    private String id;
    private String object;
    private long created;
    private String model;
    private List<OpenAIChoice> choices;
    private OpenAIUsage usage;

    public String getId() {
        return id;
    }

    public void setId(String id) {
        this.id = id;
    }

    public String getObject() {
        return object;
    }

    public void setObject(String object) {
        this.object = object;
    }

    public long getCreated() {
        return created;
    }

    public void setCreated(long created) {
        this.created = created;
    }

    public String getModel() {
        return model;
    }

    public void setModel(String model) {
        this.model = model;
    }

    public List<OpenAIChoice> getChoices() {
        return choices;
    }

    public void setChoices(List<OpenAIChoice> choices) {
        this.choices = choices;
    }

    public OpenAIUsage getUsage() {
        return usage;
    }

    public void setUsage(OpenAIUsage usage) {
        this.usage = usage;
    }
}