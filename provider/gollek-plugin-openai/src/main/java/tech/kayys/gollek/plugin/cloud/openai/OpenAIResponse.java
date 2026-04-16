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

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.List;

/**
 * OpenAI chat completions response.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public class OpenAIResponse {

    @JsonProperty("id")
    private String id;

    @JsonProperty("object")
    private String object;

    @JsonProperty("created")
    private Long created;

    @JsonProperty("model")
    private String model;

    @JsonProperty("choices")
    private List<OpenAIChoice> choices;

    @JsonProperty("usage")
    private OpenAIUsage usage;

    @JsonProperty("system_fingerprint")
    private String systemFingerprint;

    public OpenAIResponse() {
    }

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

    public Long getCreated() {
        return created;
    }

    public void setCreated(Long created) {
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

    public String getSystemFingerprint() {
        return systemFingerprint;
    }

    public void setSystemFingerprint(String systemFingerprint) {
        this.systemFingerprint = systemFingerprint;
    }

    /**
     * OpenAI choice object.
     */
    @JsonIgnoreProperties(ignoreUnknown = true)
    public static class OpenAIChoice {
        @JsonProperty("index")
        private Integer index;

        @JsonProperty("message")
        private OpenAIMessage message;

        @JsonProperty("delta")
        private OpenAIMessage delta;

        @JsonProperty("finish_reason")
        private String finishReason;

        @JsonProperty("logprobs")
        private Object logprobs;

        public Integer getIndex() {
            return index;
        }

        public void setIndex(Integer index) {
            this.index = index;
        }

        public OpenAIMessage getMessage() {
            return message;
        }

        public void setMessage(OpenAIMessage message) {
            this.message = message;
        }

        public OpenAIMessage getDelta() {
            return delta;
        }

        public void setDelta(OpenAIMessage delta) {
            this.delta = delta;
        }

        public String getFinishReason() {
            return finishReason;
        }

        public void setFinishReason(String finishReason) {
            this.finishReason = finishReason;
        }

        public Object getLogprobs() {
            return logprobs;
        }

        public void setLogprobs(Object logprobs) {
            this.logprobs = logprobs;
        }
    }
}
