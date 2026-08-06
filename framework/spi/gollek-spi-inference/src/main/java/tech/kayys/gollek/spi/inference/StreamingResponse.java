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
 *
 * @author Bhangun
 */

package tech.kayys.gollek.spi.inference;

import org.reactivestreams.Publisher;

/**
 * Streaming response wrapper
 */
public class StreamingResponse {

    private final InferenceRequest request;
    private final Publisher<InferenceResponse> responsePublisher;

    public StreamingResponse(
            InferenceRequest request,
            Publisher<InferenceResponse> responsePublisher) {
        this.request = request;
        this.responsePublisher = responsePublisher;
    }

    public InferenceRequest getRequest() {
        return request;
    }

    public Publisher<InferenceResponse> getResponsePublisher() {
        return responsePublisher;
    }

    /**
     * Builder for StreamingResponse
     */
    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private InferenceRequest request;
        private Publisher<InferenceResponse> responsePublisher;

        public Builder request(InferenceRequest request) {
            this.request = request;
            return this;
        }

        public Builder responsePublisher(Publisher<InferenceResponse> responsePublisher) {
            this.responsePublisher = responsePublisher;
            return this;
        }

        public StreamingResponse build() {
            return new StreamingResponse(request, responsePublisher);
        }
    }
}
