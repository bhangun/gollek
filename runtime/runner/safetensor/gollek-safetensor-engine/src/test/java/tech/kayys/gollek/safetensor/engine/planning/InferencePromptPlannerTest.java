/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.planning;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.spi.Message;
import tech.kayys.alkhawarizm.spi.model.ModelRuntimeTraits;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class InferencePromptPlannerTest {
    private final InferencePromptPlanner planner = new InferencePromptPlanner();

    @Test
    void injectsRuntimeTraitDefaultSystemPrompt() {
        InferencePromptPlan plan = planner.plan(
                List.of(Message.user("where is jakarta")),
                "custom",
                ModelRuntimeTraits.builder().qwenText().build(),
                false);

        assertTrue(plan.preparedPrompt().defaultSystemInjected());
        assertTrue(plan.preparedPrompt().formattedPrompt().startsWith("<|im_start|>system\nYou are Qwen"));
        assertEquals(1, occurrences(plan.preparedPrompt().formattedPrompt(), "You are Qwen"));
        assertEquals("where is jakarta", plan.ttsPrompt());
    }

    @Test
    void existingSystemMessagePreventsDefaultInjection() {
        InferencePromptPlan plan = planner.plan(
                List.of(Message.system("speak plainly"), Message.user("hello")),
                "custom",
                ModelRuntimeTraits.EMPTY,
                false);

        assertFalse(plan.preparedPrompt().defaultSystemInjected());
        assertTrue(plan.preparedPrompt().formattedPrompt().startsWith("<|im_start|>system\nspeak plainly"));
        assertFalse(plan.preparedPrompt().formattedPrompt().contains(ModelRuntimeTraits.DEFAULT_SYSTEM_PROMPT));
    }

    @Test
    void gemma4RuntimeTraitsSkipDefaultSystemInjection() {
        InferencePromptPlan plan = planner.plan(
                List.of(Message.user("where is jakarta")),
                "custom",
                ModelRuntimeTraits.builder().nativeBf16Matvec().build(),
                false);

        assertFalse(plan.preparedPrompt().defaultSystemInjected());
        assertTrue(plan.preparedPrompt().formattedPrompt().startsWith("<bos><|turn>user\nwhere is jakarta"));
    }

    @Test
    void callerCanSuppressDefaultSystemInjection() {
        InferencePromptPlan plan = planner.plan(
                List.of(Message.user("hello")),
                "custom",
                ModelRuntimeTraits.EMPTY,
                true);

        assertFalse(plan.preparedPrompt().defaultSystemInjected());
        assertFalse(plan.preparedPrompt().formattedPrompt().contains(ModelRuntimeTraits.DEFAULT_SYSTEM_PROMPT));
    }

    @Test
    void ttsPromptUsesLastUserMessage() {
        InferencePromptPlan plan = planner.plan(
                List.of(
                        Message.user("first"),
                        Message.assistant("done"),
                        Message.user("second")),
                "custom",
                ModelRuntimeTraits.EMPTY,
                false);

        assertEquals("second", plan.ttsPrompt());
    }

    private static int occurrences(String value, String token) {
        int count = 0;
        int index = 0;
        while ((index = value.indexOf(token, index)) >= 0) {
            count++;
            index += token.length();
        }
        return count;
    }
}
