/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 */

package tech.kayys.gollek.plugin.core;

import java.util.List;
import java.util.Map;

public final class TestClassLoaderExtensionAvailabilityProvider implements ExtensionAvailabilityProvider {
    public static final String ID = "classloader-test";

    @Override
    public String extensionId() {
        return ID;
    }

    @Override
    public String extensionName() {
        return "Classloader Test";
    }

    @Override
    public String extensionKind() {
        return "tokenizer";
    }

    @Override
    public ExtensionAvailability availability() {
        return new ExtensionAvailability(
                ID,
                "Classloader Test",
                extensionKind(),
                true,
                false,
                true,
                true,
                "ready",
                List.of("tokenizer"),
                List.of("sentencepiece"),
                Map.of("source", "classloader"),
                "loaded through an explicit classloader",
                List.of());
    }
}
