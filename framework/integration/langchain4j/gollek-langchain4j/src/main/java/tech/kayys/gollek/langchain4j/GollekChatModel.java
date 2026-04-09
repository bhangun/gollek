package tech.kayys.gollek.langchain4j;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.data.message.TextContent;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.output.Response;
import tech.kayys.gollek.sdk.GollekClient;
import tech.kayys.gollek.sdk.model.GenerationRequest;

import java.util.List;
import java.util.Objects;

/**
 * Gollek implementation of LangChain4j ChatLanguageModel.
 */
public class GollekChatModel implements ChatLanguageModel {

    private final GollekClient client;
    private final String model;
    private final float temperature;
    private final int maxTokens;

    private GollekChatModel(Builder builder) {
        this.client = Objects.requireNonNull(builder.client, "client is required");
        this.model = builder.model;
        this.temperature = builder.temperature;
        this.maxTokens = builder.maxTokens;
    }

    public static Builder builder() {
        return new Builder();
    }

    @Override
    public Response<AiMessage> generate(List<ChatMessage> messages) {
        // Check if there are multimodal parts
        boolean isMultimodal = messages.stream()
                .anyMatch(m -> m instanceof UserMessage um && um.contents() != null && 
                        um.contents().stream().anyMatch(c -> !(c instanceof TextContent)));

        if (isMultimodal) {
            var parts = GollekMessageMapper.toParts(messages);
            var result = tech.kayys.gollek.ml.Gollek.multimodal(model)
                    .parts(parts)
                    .generate();
            return Response.from(AiMessage.from(result.text()));
        }

        // Standard text generation
        String prompt = GollekMessageMapper.toPrompt(messages);

        var request = GenerationRequest.builder()
                .prompt(prompt)
                .model(model)
                .temperature(temperature)
                .maxTokens(maxTokens)
                .build();

        var response = client.generate(request);

        return Response.from(AiMessage.from(response.text()));
    }

    public static class Builder {
        private GollekClient client;
        private String model;
        private float temperature = 0.7f;
        private int maxTokens = 512;

        public Builder client(GollekClient client) {
            this.client = client;
            return this;
        }

        public Builder endpoint(String endpoint) {
            this.client = GollekClient.builder().endpoint(endpoint).build();
            return this;
        }

        public Builder model(String model) {
            this.model = model;
            return this;
        }

        public Builder temperature(double temperature) {
            this.temperature = (float) temperature;
            return this;
        }

        public Builder maxTokens(int maxTokens) {
            this.maxTokens = maxTokens;
            return this;
        }

        public GollekChatModel build() {
            return new GollekChatModel(this);
        }
    }
}
