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
 * OpenAI embedding response.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public class OpenAIEmbeddingResponse {

    @JsonProperty("object")
    private String object;

    @JsonProperty("data")
    private List<OpenAIEmbeddingData> data;

    @JsonProperty("model")
    private String model;

    @JsonProperty("usage")
    private OpenAIUsage usage;

    public OpenAIEmbeddingResponse() {
    }

    public String getObject() {
        return object;
    }

    public void setObject(String object) {
        this.object = object;
    }

    public List<OpenAIEmbeddingData> getData() {
        return data;
    }

    public void setData(List<OpenAIEmbeddingData> data) {
        this.data = data;
    }

    public String getModel() {
        return model;
    }

    public void setModel(String model) {
        this.model = model;
    }

    public OpenAIUsage getUsage() {
        return usage;
    }

    public void setUsage(OpenAIUsage usage) {
        this.usage = usage;
    }
}
