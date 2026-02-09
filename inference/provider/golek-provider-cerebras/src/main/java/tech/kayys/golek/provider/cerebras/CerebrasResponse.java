package tech.kayys.golek.provider.cerebras;

import java.util.List;

/**
 * Cerebras response DTO
 */
public class CerebrasResponse {

    private String id;
    private String object;
    private long created;
    private String model;
    private List<CerebrasChoice> choices;
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

    public List<CerebrasChoice> getChoices() {
        return choices;
    }

    public void setChoices(List<CerebrasChoice> choices) {
        this.choices = choices;
    }

    public CerebrasUsage getUsage() {
        return usage;
    }

    public void setUsage(CerebrasUsage usage) {
        this.usage = usage;
    }
}