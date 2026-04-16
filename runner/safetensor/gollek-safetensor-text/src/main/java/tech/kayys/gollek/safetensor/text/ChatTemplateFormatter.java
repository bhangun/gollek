/*
 * Gollek Inference Engine — SafeTensor Text Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * ChatTemplateFormatter.java
 */
package tech.kayys.gollek.safetensor.text;

import java.util.List;
import java.util.Locale;
import tech.kayys.gollek.spi.Message;

public final class ChatTemplateFormatter {

    private ChatTemplateFormatter() {}

    public static String format(List<Message> messages, String modelType) {
        if (modelType == null) modelType = "";
        return switch (normalise(modelType)) {
            case "llama3", "llama3.1", "llama3.2", "llama-3" -> formatLlama3(messages);
            case "mistral", "mixtral"                          -> formatMistral(messages);
            case "phi3", "phi-3", "phi3_5", "phi3.5", "phi"  -> formatPhi3(messages);
            case "gemma", "gemma2"                             -> formatGemma(messages);
            case "qwen2", "qwen2.5", "qwen"                   -> formatQwen(messages);
            default                                            -> formatChatML(messages);
        };
    }

    private static String formatLlama3(List<Message> messages) {
        var sb = new StringBuilder("<|begin_of_text|>\n");
        for (var m : messages) {
            String roleName = m.getRole().name().toLowerCase(Locale.ROOT);
            sb.append("<|start_header_id|>").append(roleName).append("<|end_header_id|>\n\n");
            sb.append(m.getContent()).append("<|eot_id|>\n");
        }
        sb.append("<|start_header_id|>assistant<|end_header_id|>\n\n");
        return sb.toString();
    }

    private static String formatMistral(List<Message> messages) {
        var sb = new StringBuilder();
        String sys = extractSystem(messages);
        boolean first = true;
        for (var m : messages) {
            if (m.getRole() == Message.Role.SYSTEM) continue;
            if (m.getRole() == Message.Role.USER) {
                sb.append("[INST] ");
                if (first && sys != null) {
                    sb.append(sys).append("\n\n");
                    first = false;
                }
                sb.append(m.getContent()).append(" [/INST]");
            } else if (m.getRole() == Message.Role.ASSISTANT) {
                sb.append(m.getContent()).append("</s>");
            }
        }
        return sb.toString();
    }

    private static String formatPhi3(List<Message> messages) {
        var sb = new StringBuilder();
        for (var m : messages) {
            String roleName = m.getRole().name().toLowerCase(Locale.ROOT);
            sb.append("<|").append(roleName).append("|>\n");
            sb.append(m.getContent()).append("<|end|>\n");
        }
        sb.append("<|assistant|>\n");
        return sb.toString();
    }

    private static String formatGemma(List<Message> messages) {
        var sb = new StringBuilder();
        for (var m : messages) {
            String roleName = m.getRole() == Message.Role.SYSTEM ? "user" : m.getRole().name().toLowerCase(Locale.ROOT);
            sb.append("<start_of_turn>").append(roleName).append("\n");
            sb.append(m.getContent()).append("<end_of_turn>\n");
        }
        sb.append("<start_of_turn>model\n");
        return sb.toString();
    }

    private static String formatQwen(List<Message> messages) {
        return formatChatML(messages);
    }

    private static String formatChatML(List<Message> messages) {
        var sb = new StringBuilder();
        for (var m : messages) {
            String roleName = m.getRole().name().toLowerCase(Locale.ROOT);
            sb.append("<|im_start|>").append(roleName).append("\n");
            sb.append(m.getContent()).append("<|im_end|>\n");
        }
        sb.append("<|im_start|>assistant\n");
        return sb.toString();
    }

    private static String extractSystem(List<Message> messages) {
        for (var m : messages) {
            if (m.getRole() == Message.Role.SYSTEM) return m.getContent();
        }
        return null;
    }

    private static String normalise(String s) {
        return s.trim().toLowerCase(Locale.ROOT);
    }
}
