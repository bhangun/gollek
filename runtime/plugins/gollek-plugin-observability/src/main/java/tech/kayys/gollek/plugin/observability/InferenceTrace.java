/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 *
 * @author Bhangun
 */

package tech.kayys.gollek.plugin.observability;

import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Trace model capturing inference execution details.
 * Records phases, durations, token counts, tool usage, and errors.
 */
public class InferenceTrace {

    private final String requestId;
    private final Instant startTime;
    private int inputTokens;
    private int outputTokens;
    private int reasoningSteps;
    private String error;
    private final List<String> toolsUsed = new ArrayList<>();
    private final Map<String, Duration> phaseLatencies = new HashMap<>();

    public InferenceTrace(String requestId, Instant startTime) {
        this.requestId = requestId;
        this.startTime = startTime;
    }

    public String getRequestId() {
        return requestId;
    }

    public Instant getStartTime() {
        return startTime;
    }

    public int getInputTokens() {
        return inputTokens;
    }

    public void setInputTokens(int inputTokens) {
        this.inputTokens = inputTokens;
    }

    public int getOutputTokens() {
        return outputTokens;
    }

    public void setOutputTokens(int outputTokens) {
        this.outputTokens = outputTokens;
    }

    public int getReasoningSteps() {
        return reasoningSteps;
    }

    public void setReasoningSteps(int reasoningSteps) {
        this.reasoningSteps = reasoningSteps;
    }

    public String getError() {
        return error;
    }

    public void setError(String error) {
        this.error = error;
    }

    public List<String> getToolsUsed() {
        return toolsUsed;
    }

    public void addToolUsage(String toolId) {
        toolsUsed.add(toolId);
    }

    public Map<String, Duration> getPhaseLatencies() {
        return phaseLatencies;
    }

    public void addPhaseLatency(String phase, Duration duration) {
        phaseLatencies.put(phase, duration);
    }

    public int getTotalTokens() {
        return inputTokens + outputTokens;
    }

    public boolean hasError() {
        return error != null;
    }

    @Override
    public String toString() {
        return "InferenceTrace{" +
                "requestId='" + requestId + '\'' +
                ", inputTokens=" + inputTokens +
                ", outputTokens=" + outputTokens +
                ", reasoningSteps=" + reasoningSteps +
                ", toolsUsed=" + toolsUsed.size() +
                ", hasError=" + hasError() +
                '}';
    }
}
