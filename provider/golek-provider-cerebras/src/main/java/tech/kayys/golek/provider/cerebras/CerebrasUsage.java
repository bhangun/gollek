package tech.kayys.golek.provider.cerebras;

import java.time.Duration;
import java.util.Optional;

import java.time.Instant;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.List;

public class CerebrasUsage {

    @JsonProperty("prompt_tokens")
    private Integer promptTokens;

    @JsonProperty("completion_tokens")
    private Integer completionTokens;

    @JsonProperty("total_tokens")
    private Integer totalTokens;

    public Integer getPromptTokens() { return promptTokens; 
}