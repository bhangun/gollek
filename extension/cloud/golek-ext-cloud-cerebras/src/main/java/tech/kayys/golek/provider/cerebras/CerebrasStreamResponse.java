package tech.kayys.golek.provider.cerebras;

import java.util.List;

/**
 * Cerebras streaming response DTO
 */
public class CerebrasStreamResponse {

    private String id;
    private String object;
    private long created;
    private String model;
    private List<CerebrasStreamChoice> choices;
    private CerebrasUsage usage;

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

    public List<CerebrasStreamChoice> getChoices() {
        return choices;
    }

    public void setChoices(List<CerebrasStreamChoice> choices) {
        this.choices = choices;
    }

    public CerebrasUsage getUsage() {
        return usage;
    }

    public void setUsage(CerebrasUsage usage) {
        this.usage = usage;
    }
}
