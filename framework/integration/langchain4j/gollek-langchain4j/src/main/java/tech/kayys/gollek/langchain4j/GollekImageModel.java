package tech.kayys.gollek.langchain4j;

import dev.langchain4j.data.image.Image;
import dev.langchain4j.model.image.ImageModel;
import dev.langchain4j.model.output.Response;
import tech.kayys.gollek.ml.Gollek;
import tech.kayys.gollek.sdk.multimodal.MultimodalResult;

import java.util.Collections;
import java.util.List;
import java.util.Base64;

/**
 * Gollek implementation of LangChain4j ImageModel for text-to-image generation.
 */
public class GollekImageModel implements ImageModel {

    private final String model;
    private final String quality;
    private final String size;

    private GollekImageModel(Builder builder) {
        this.model = builder.model;
        this.quality = builder.quality;
        this.size = builder.size;
    }

    public static Builder builder() {
        return new Builder();
    }

    @Override
    public Response<Image> generate(String prompt) {
        // Use the new Gollek fluent API for vision/image generation
        MultimodalResult result = Gollek.vision(model)
                .prompt(prompt)
                .generate();

        // Check if we have binary data (image)
        if (result.hasBinary()) {
            byte[] data = result.binaryData();
            String base64 = Base64.getEncoder().encodeToString(data);
            
            // For now, returning as base64 data. 
            // LangChain4j Image can also take a URL or Path.
            Image image = Image.builder()
                    .base64Data(base64)
                    .build();
            
            return Response.from(image);
        }

        throw new RuntimeException("Gollek failed to generate an image for the given prompt. " +
                "Response text: " + result.text());
    }

    @Override
    public Response<List<Image>> generate(String prompt, int n) {
        // Simple implementation: generate n images sequentially
        // In a production backend, this might be a single batch request
        return Response.from(Collections.singletonList(generate(prompt).content()));
    }

    public static class Builder {
        private String model = "stable-diffusion-xl";
        private String quality = "standard";
        private String size = "1024x1024";

        public Builder model(String model) {
            this.model = model;
            return this;
        }

        public Builder quality(String quality) {
            this.quality = quality;
            return this;
        }

        public Builder size(String size) {
            this.size = size;
            return this;
        }

        public GollekImageModel build() {
            return new GollekImageModel(this);
        }
    }
}
