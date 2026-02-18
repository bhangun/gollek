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

package tech.kayys.gollek.spi.plugin;

import tech.kayys.gollek.spi.tool.ToolCall;

import java.util.List;
import java.util.Optional;

/**
 * SPI for plugins that implement multi-turn reasoning loops.
 * <p>
 * Responsible for:
 * <ul>
 * <li>Iterative reasoning loop control</li>
 * <li>Stop condition detection (final answer, tool call, max steps)</li>
 * <li>Tool call parsing from LLM output</li>
 * <li>Self-repair of malformed JSON/tool calls</li>
 * <li>Retry on malformed output</li>
 * </ul>
 * <p>
 * This implements the LLM → tool call → execute → feed back → LLM → final
 * answer loop.
 * llama.cpp / LiteRT have no concept of multi-turn reasoning.
 */
public interface ReasoningPlugin extends GollekPlugin {

    /**
     * Determine whether the reasoning loop should continue based on the latest
     * LLM output.
     *
     * @param output      the raw LLM output text
     * @param currentStep the current step number (0-indexed)
     * @return true if the loop should continue (e.g., tool call detected), false
     *         for final answer
     */
    boolean shouldContinue(String output, int currentStep);

    /**
     * Parse tool calls from LLM output.
     * Returns empty list if no tool calls are detected.
     *
     * @param output the raw LLM output text
     * @return list of detected tool calls, empty if none
     */
    List<ToolCall> parseToolCalls(String output);

    /**
     * Attempt to repair malformed LLM output (e.g., broken JSON, incomplete tool
     * calls).
     *
     * @param malformedOutput the malformed output
     * @return repaired output, or empty if repair is not possible
     */
    Optional<String> repairMalformedOutput(String malformedOutput);

    /**
     * Extract the final answer from the LLM output, stripping any reasoning traces
     * or intermediate content.
     *
     * @param output the raw LLM output that contains a final answer
     * @return the clean final answer text
     */
    String extractFinalAnswer(String output);

    /**
     * Maximum number of reasoning steps before forced termination.
     * Default: 10
     */
    default int maxSteps() {
        return 10;
    }

    /**
     * Maximum number of repair attempts for malformed output per step.
     * Default: 2
     */
    default int maxRepairAttempts() {
        return 2;
    }

    /**
     * Reasoning loop result.
     *
     * @param finalAnswer the final answer text
     * @param toolCalls   accumulated tool calls (empty if direct answer)
     * @param steps       number of reasoning steps taken
     * @param wasRepaired whether any output repair was needed
     */
    record ReasoningResult(
            String finalAnswer,
            List<ToolCall> toolCalls,
            int steps,
            boolean wasRepaired) {
    }
}
