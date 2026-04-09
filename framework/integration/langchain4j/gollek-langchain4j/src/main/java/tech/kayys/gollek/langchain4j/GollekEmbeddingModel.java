package tech.kayys.gollek.langchain4j;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.output.Response;
import tech.kayys.gollek.sdk.GollekClient;
import tech.kayys.gollek.sdk.model.EmbeddingRequest;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Gollek implementation of LangChain4j EmbeddingModel.
 */
public class GollekEmbeddingModel implements EmbeddingModel {

    private final GollekClient client;
    private final String model;

    private GollekEmbeddingModel(Builder builder) {
        this.client = Objects.requireNonNull(builder.client, "client is required");
        this.model = builder.model;
    }

    public static Builder builder() {
        return new Builder();
    }

    @Override
    public Response<List<Embedding>> embedAll(List<TextSegment> textSegments) {
        List<Embedding> embeddings = new ArrayList<>();
        for (TextSegment segment : textSegments) {
            var request = EmbeddingRequest.builder()
                    .input(segment.text())
                    .model(model)
                    .build();
            var response = client.embed(request);
            embeddings.add(Embedding.from(response.vector()));
        }
        return Response.from(embeddings);
    }

    public static class Builder {
        private GollekClient client;
        private String model;

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

        public GollekEmbeddingModel build() {
            return new GollekEmbeddingModel(this);
        }
    }
}
