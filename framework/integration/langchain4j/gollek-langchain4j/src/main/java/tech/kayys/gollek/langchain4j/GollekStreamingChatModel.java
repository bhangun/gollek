package tech.kayys.gollek.langchain4j;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.model.StreamingResponseHandler;
import dev.langchain4j.model.chat.StreamingChatLanguageModel;
import dev.langchain4j.model.output.Response;
import tech.kayys.gollek.sdk.GollekClient;
import tech.kayys.gollek.sdk.model.GenerationRequest;

import java.util.List;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Gollek implementation of LangChain4j StreamingChatLanguageModel.
 */
public class GollekStreamingChatModel implements StreamingChatLanguageModel {

    private final GollekClient client;
    private final String model;
    private final float temperature;
    private final int maxTokens;

    private GollekStreamingChatModel(Builder builder) {
        this.client = Objects.requireNonNull(builder.client, "client is required");
        this.model = builder.model;
        this.temperature = builder.temperature;
        this.maxTokens = builder.maxTokens;
    }

    public static Builder builder() {
        return new Builder();
    }

    @Override
    public void generate(List<ChatMessage> messages, StreamingResponseHandler<AiMessage> handler) {
        String prompt = GollekMessageMapper.toPrompt(messages);

        var request = GenerationRequest.builder()
                .prompt(prompt)
                .temperature(temperature)
                .maxTokens(maxTokens)
                .stream(true)
                .build();

        StringBuilder fullText = new StringBuilder();

        client.generateStream(request)
                .onToken(token -> {
                    fullText.append(token);
                    handler.onNext(token);
                })
                .onComplete(() -> {
                    Response<AiMessage> response = Response.from(AiMessage.from(fullText.toString()));
                    handler.onComplete(response);
                })
                .onError(handler::onError);
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

        public GollekStreamingChatModel build() {
            return new GollekStreamingChatModel(this);
        }
    }
}
