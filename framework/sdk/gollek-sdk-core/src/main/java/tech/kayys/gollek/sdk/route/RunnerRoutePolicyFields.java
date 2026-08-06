/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 */

package tech.kayys.gollek.sdk.route;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Stable runner-selection policy vocabulary shared by CLI route commands and CI reports.
 */
public final class RunnerRoutePolicyFields {
    public static final String CONTRACT_ID = "gollek.runner-route.policy";
    public static final int SCHEMA_VERSION = 1;

    public static final String AUTO = "auto";
    public static final String HYBRID = "hybrid";
    public static final String SAFETENSOR = "safetensor";
    public static final String GGUF = "gguf";
    public static final String LITERT = "litert";

    private RunnerRoutePolicyFields() {
    }

    public static Map<String, Object> report() {
        Map<String, Object> report = new LinkedHashMap<>();
        report.put("contractId", CONTRACT_ID);
        report.put("schemaVersion", SCHEMA_VERSION);
        report.put("schemaFingerprint", schemaFingerprint());
        report.put("defaultRunner", AUTO);
        report.put("supportedRunners", supportedRunners());
        report.put("aliases", runnerAliases());
        report.put("effects", effects());
        report.put("providerConflictPolicy",
                "fixed runners reject conflicting explicit --provider values");
        report.put("autoDetectionPolicy",
                "omitted --runner preserves provider and format inputs for repository/provider auto detection");
        return report;
    }

    public static List<String> supportedRunners() {
        return List.of(AUTO, HYBRID, SAFETENSOR, GGUF, LITERT);
    }

    public static Map<String, List<String>> runnerAliases() {
        Map<String, List<String>> aliases = new LinkedHashMap<>();
        aliases.put(SAFETENSOR, List.of("safe-tensor", "safe-tensors", "safetensors"));
        aliases.put(LITERT, List.of("lite-rt", "lite_rt", "tflite", "task", "litertlm"));
        aliases.put(HYBRID, List.of("mixed", "fallback", "auto-hybrid", "auto_hybrid"));
        return Collections.unmodifiableMap(aliases);
    }

    public static List<Map<String, Object>> effects() {
        return List.of(
                effect(AUTO, false, null, null, false, false),
                effect(HYBRID, true, null, null, true, false),
                effect(SAFETENSOR, true, "safetensor", "safetensors", false, false),
                effect(GGUF, true, "gguf", "gguf", false, true),
                effect(LITERT, true, "litert", "litert", false, false));
    }

    public static String schemaFingerprint() {
        String payload = String.join("\n",
                "contractId=" + CONTRACT_ID,
                "schemaVersion=" + SCHEMA_VERSION,
                "defaultRunner=" + AUTO,
                "supportedRunners=" + String.join(",", supportedRunners()),
                "aliases=" + runnerAliases().entrySet(),
                "effects=" + effects());
        return "sha256:" + sha256(payload);
    }

    private static Map<String, Object> effect(
            String runner,
            boolean explicit,
            String provider,
            String format,
            boolean preferAlternateRuntime,
            boolean forceGguf) {
        Map<String, Object> effect = new LinkedHashMap<>();
        effect.put("runner", runner);
        effect.put("explicit", explicit);
        effect.put("provider", provider);
        effect.put("format", format);
        effect.put("preferAlternateRuntime", preferAlternateRuntime);
        effect.put("forceGguf", forceGguf);
        return effect;
    }

    private static String sha256(String value) {
        try {
            byte[] digest = MessageDigest.getInstance("SHA-256")
                    .digest(value.getBytes(StandardCharsets.UTF_8));
            StringBuilder hex = new StringBuilder(digest.length * 2);
            for (byte raw : digest) {
                int valueByte = raw & 0xff;
                if (valueByte < 16) {
                    hex.append('0');
                }
                hex.append(Integer.toHexString(valueByte));
            }
            return hex.toString();
        } catch (NoSuchAlgorithmException error) {
            throw new IllegalStateException("SHA-256 is required for runner policy fingerprints.", error);
        }
    }
}
