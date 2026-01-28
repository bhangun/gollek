package tech.kayys.golek.provider.openai;

import com.fasterxml.jackson.annotation.JsonInclude;

import java.util.List;

/**
 * OpenAI embedding request
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
public class OpenAIEmbeddingRequest {

    private String model;
    private Object input; // String or List<String>
    private String user;
    private String encoding_format;

    public String getModel() {
        return model;
    }

    public void setModel(String model) {
        this.model = model;
    }

    public Object getInput() {
        return input;
    }

    public void setInput(Object input) {
        this.input = input;
    }

    public String getUser() {
        return user;
    }

    public void setUser(String user) {
        this.user = user;
    }

    public String getEncoding_format() {
        return encoding_format;
    }

    public void setEncoding_format(String encoding_format) {
        this.encoding_format = encoding_format;
    }
}
