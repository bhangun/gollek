/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
 */

package tech.kayys.gollek.plugin.cloud.openai;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import org.jboss.logging.Logger;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Executor;

/**
 * HTTP client for OpenAI API communication.
 */
public class OpenAIClient {

    private static final Logger LOG = Logger.getLogger(OpenAIClient.class);

    private final HttpClient httpClient;
    private final ObjectMapper objectMapper;
    private final String baseUrl;
    private final String apiKey;
    private final String organization;
    private final Duration timeout;

    /**
     * Create a new OpenAI client.
     *
     * @param config OpenAI configuration
     */
    public OpenAIClient(OpenAiConfig config) {
        this.objectMapper = new ObjectMapper();
        this.baseUrl = config.getBaseUrlNormalized();
        this.apiKey = config.apiKey();
        this.organization = config.organization();
        this.timeout = Duration.ofSeconds(config.timeoutSeconds());

        this.httpClient = HttpClient.newBuilder()
                .connectTimeout(timeout)
                .build();
    }

    /**
     * Create a new OpenAI client with custom executor.
     *
     * @param config OpenAI configuration
     * @param executor Custom executor for async operations
     */
    public OpenAIClient(OpenAiConfig config, Executor executor) {
        this.objectMapper = new ObjectMapper();
        this.baseUrl = config.getBaseUrlNormalized();
        this.apiKey = config.apiKey();
        this.organization = config.organization();
        this.timeout = Duration.ofSeconds(config.timeoutSeconds());

        this.httpClient = HttpClient.newBuilder()
                .connectTimeout(timeout)
                .executor(executor)
                .build();
    }

    /**
     * Send a chat completion request.
     *
     * @param request Chat completion request
     * @return Uni with the response
     */
    public Uni<OpenAIResponse> chatCompletions(OpenAIRequest request) {
        return Uni.createFrom().completionStage(() -> {
            try {
                String body = objectMapper.writeValueAsString(request);
                HttpRequest httpRequest = buildRequest("/chat/completions", body, false);

                return httpClient.sendAsync(httpRequest, HttpResponse.BodyHandlers.ofString());
            } catch (Exception e) {
                throw new RuntimeException("Failed to build chat completions request", e);
            }
        })
        .map(response -> {
            if (response.statusCode() != 200) {
                throw new OpenAiApiException("OpenAI API error: " + response.statusCode() + " - " + response.body());
            }
            try {
                return objectMapper.readValue(response.body(), OpenAIResponse.class);
            } catch (Exception e) {
                throw new RuntimeException("Failed to parse OpenAI response", e);
            }
        });
    }

    /**
     * Send a streaming chat completion request.
     *
     * @param request Chat completion request with stream=true
     * @return Multi with streaming response chunks
     */
    public Multi<OpenAIStreamResponse> chatCompletionsStream(OpenAIRequest request) {
        request.setStream(true);

        return Multi.createFrom().emitter(emitter -> {
            try {
                String body = objectMapper.writeValueAsString(request);
                HttpRequest httpRequest = buildRequest("/chat/completions", body, true);

                httpClient.sendAsync(httpRequest, HttpResponse.BodyHandlers.ofInputStream())
                        .thenAccept(response -> {
                            if (response.statusCode() != 200) {
                                emitter.fail(new OpenAiApiException(
                                        "OpenAI streaming error: " + response.statusCode()));
                                return;
                            }

                            try (BufferedReader reader = new BufferedReader(
                                    new InputStreamReader(response.body()))) {
                                String line;
                                while ((line = reader.readLine()) != null) {
                                    if (line.startsWith("data: ")) {
                                        String data = line.substring(6).trim();
                                        if (data.isEmpty() || "[DONE]".equals(data)) {
                                            continue;
                                        }

                                        try {
                                            OpenAIStreamResponse chunk = objectMapper.readValue(
                                                    data, OpenAIStreamResponse.class);
                                            emitter.emit(chunk);
                                        } catch (Exception e) {
                                            LOG.warnf("Failed to parse streaming chunk: %s", data);
                                        }
                                    }
                                }
                                emitter.complete();
                            } catch (Exception e) {
                                emitter.fail(e);
                            }
                        })
                        .exceptionally(t -> {
                            emitter.fail(t);
                            return null;
                        });

            } catch (Exception e) {
                emitter.fail(e);
            }
        });
    }

    /**
     * Build HTTP request with proper headers.
     *
     * @param path API path
     * @param body Request body
     * @param streaming Whether this is a streaming request
     * @return HTTP request
     */
    private HttpRequest buildRequest(String path, String body, boolean streaming) {
        HttpRequest.Builder builder = HttpRequest.newBuilder()
                .uri(URI.create(baseUrl + path))
                .timeout(timeout)
                .header("Content-Type", "application/json")
                .header("Authorization", "Bearer " + apiKey);

        if (organization != null && !organization.isBlank()) {
            builder.header("OpenAI-Organization", organization);
        }

        if (streaming) {
            builder.header("Accept", "text/event-stream");
        }

        builder.POST(HttpRequest.BodyPublishers.ofString(body));

        return builder.build();
    }

    /**
     * Send an embedding request.
     *
     * @param request Embedding request
     * @return Uni with the response
     */
    public Uni<OpenAIEmbeddingResponse> embeddings(OpenAIEmbeddingRequest request) {
        return Uni.createFrom().completionStage(() -> {
            try {
                String body = objectMapper.writeValueAsString(request);
                HttpRequest httpRequest = buildRequest("/embeddings", body, false);

                return httpClient.sendAsync(httpRequest, HttpResponse.BodyHandlers.ofString());
            } catch (Exception e) {
                throw new RuntimeException("Failed to build embeddings request", e);
            }
        })
        .map(response -> {
            if (response.statusCode() != 200) {
                throw new OpenAiApiException("OpenAI API error: " + response.statusCode() + " - " + response.body());
            }
            try {
                return objectMapper.readValue(response.body(), OpenAIEmbeddingResponse.class);
            } catch (Exception e) {
                throw new RuntimeException("Failed to parse OpenAI embedding response", e);
            }
        });
    }

    /**
     * OpenAI API exception.
     */
    public static class OpenAiApiException extends RuntimeException {
        public OpenAiApiException(String message) {
            super(message);
        }

        public OpenAiApiException(String message, Throwable cause) {
            super(message, cause);
        }
    }
}
