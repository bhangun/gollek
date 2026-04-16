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

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * OpenAI tool definition.
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
public class OpenAITool {

    @JsonProperty("type")
    private String type = "function";

    @JsonProperty("function")
    private OpenAIFunction function;

    public OpenAITool() {
    }

    public OpenAITool(OpenAIFunction function) {
        this.type = "function";
        this.function = function;
    }

    public String getType() {
        return type;
    }

    public void setType(String type) {
        this.type = type;
    }

    public OpenAIFunction getFunction() {
        return function;
    }

    public void setFunction(OpenAIFunction function) {
        this.function = function;
    }
}
