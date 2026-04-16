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

/**
 * OpenAI streaming choice.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public class OpenAIStreamChoice {

    @JsonProperty("index")
    private Integer index;

    @JsonProperty("delta")
    private OpenAIMessage delta;

    @JsonProperty("message")
    private OpenAIMessage message;

    @JsonProperty("finish_reason")
    private String finishReason;

    @JsonProperty("logprobs")
    private Object logprobs;

    public OpenAIStreamChoice() {
    }

    public Integer getIndex() {
        return index;
    }

    public void setIndex(Integer index) {
        this.index = index;
    }

    public OpenAIMessage getDelta() {
        return delta;
    }

    public void setDelta(OpenAIMessage delta) {
        this.delta = delta;
    }

    public OpenAIMessage getMessage() {
        return message;
    }

    public void setMessage(OpenAIMessage message) {
        this.message = message;
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
