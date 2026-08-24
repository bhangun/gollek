/*
 * Gollek CLI
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.route;
import tech.kayys.gollek.sdk.route.*;

import tech.kayys.gollek.sdk.route.RoutePreflightDiagnosticFields.ProblemDetail;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;
import tech.kayys.alkhawarizm.spi.model.ModelFamilyQuantizedLoaderProfile;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.function.Predicate;

/**
 * Route policy for legacy direct safetensor execution and its local fallback
 * artifacts.
 */
public final class DirectSafetensorRoutePolicy {

    public static LocalModelResolver resolver;

    private static final String DEFAULT_RUN_SYSTEM_PROMPT = "Answer directly and briefly.";
    private static final String QWEN_SAFETENSOR_RUN_SYSTEM_PROMPT = "You are a helpful assistant.";
    private static final String DISABLE_DEFAULT_RUN_SYSTEM_PROPERTY = "gollek.cli.disable_default_run_system";
    private static final String ENABLE_GEMMA3_ALTERNATE_RUNTIME_PROPERTY =
            "gollek.cli.enable_gemma3_alternate_runtime";
    public static final String GEMMA4_MOBILE_QAT_LITERT_CACHE_DIR_PROPERTY =
            "gollek.cli.gemma4_mobile_qat_litert_cache_dir";
    public static final String GEMMA4_MOBILE_QAT_LITERT_CACHE_ENABLED_PROPERTY =
            "gollek.cli.gemma4_mobile_qat_litert_cache_enabled";
    public static final String GEMMA4_TEXT_GGUF_CACHE_DIR_PROPERTY =
            "gollek.cli.gemma4_text_gguf_cache_dir";
    public static final String GEMMA4_TEXT_GGUF_CACHE_ENABLED_PROPERTY =
            "gollek.cli.gemma4_text_gguf_cache_enabled";
    public static final String COMMUNITY_TEXT_GGUF_CACHE_DIR_PROPERTY =
            "gollek.cli.community_text_gguf_cache_dir";
    public static final String COMMUNITY_TEXT_GGUF_CACHE_ENABLED_PROPERTY =
            "gollek.cli.community_text_gguf_cache_enabled";
    public static final String GEMMA4_MOBILE_QAT_LITERT_CACHE_KIND = "gemma4_mobile_qat_litert";
    public static final String GEMMA4_TEXT_GGUF_CACHE_KIND = "gemma4_text_gguf";
    public static final String COMMUNITY_TEXT_GGUF_CACHE_KIND = "community_text_gguf";
    private static final String GEMMA4_MOBILE_QAT_LITERT_CACHE_FILE = "gemma4-mobile-qat-litert-routes.tsv";
    private static final String GEMMA4_TEXT_GGUF_CACHE_FILE = "gemma4-text-gguf-routes.tsv";
    private static final String COMMUNITY_TEXT_GGUF_CACHE_FILE = "community-text-gguf-routes.tsv";
    private static final String[] LITERT_SUFFIXES = { ".litertlm", ".tflite", ".task" };
    private static final String[] GGUF_SUFFIXES = { ".gguf" };

    private DirectSafetensorRoutePolicy() {
    }

    public record AlternateRuntimeSelection(
            String provider,
            String localPath,
            String format,
            String notice,
            boolean cacheHit,
            String cacheKind) {
        public static AlternateRuntimeSelection none() {
            return new AlternateRuntimeSelection(null, null, null, null, false, null);
        }

        public boolean selected() {
            return provider != null && !provider.isBlank()
                    && localPath != null && !localPath.isBlank();
        }

        public boolean hasNotice() {
            return notice != null && !notice.isBlank();
        }

        public String reason() {
            if (hasNotice()) {
                return notice;
            }
            if (selected()) {
                return "Direct safetensor route policy selected alternate " + provider + " runtime artifact.";
            }
            return null;
        }
    }

    public record RouteValidation(boolean allowed, List<String> messages, Map<String, Object> details) {
        public RouteValidation {
            messages = messages == null ? List.of() : List.copyOf(messages);
            details = details == null ? Map.of() : Map.copyOf(details);
        }

        public static RouteValidation pass() {
            return new RouteValidation(true, List.of(), Map.of());
        }

        public static RouteValidation invalid(List<String> messages) {
            return invalid(messages, Map.of());
        }

        public static RouteValidation invalid(List<String> messages, Map<String, Object> details) {
            return new RouteValidation(false, messages, details);
        }
    }

    public static boolean shouldUseDirectRun(String currentProvider, Path checkpointPath, boolean safetensorCheckpointDir,
            String sessionId, String grammar, String prompt, boolean directInferenceEngineAvailable) {
        if (!"safetensor".equalsIgnoreCase(currentProvider)) {
            return false;
        }
        if (checkpointPath == null || !safetensorCheckpointDir) {
            return false;
        }
        DirectSafetensorRunProfile profile = DirectSafetensorRunProfile.load(checkpointPath);
        if (profile.nativeBf16Matvec() && !allowGuardedGemma4TextDirectRun(checkpointPath, profile)) {
            return false;
        }
        if (hasText(sessionId) || hasText(grammar)) {
            return false;
        }
        return hasText(prompt) && directInferenceEngineAvailable;
    }

    public static DirectSafetensorRunProfile profileForProvider(String providerId, Path checkpointPath) {
        if (!"safetensor".equalsIgnoreCase(providerId) || checkpointPath == null) {
            return DirectSafetensorRunProfile.unresolved();
        }
        return DirectSafetensorRunProfile.load(checkpointPath);
    }

    private static boolean allowGuardedGemma4TextDirectRun(Path checkpointPath, DirectSafetensorRunProfile profile) {
        if (checkpointPath == null || profile == null || !profile.nativeBf16Matvec()) {
            return false;
        }
        if (!isGemma4Unified(profile) || isGemma4MobileQat(checkpointPath)) {
            return false;
        }
        return validateGemma4UnifiedExecutionRoute(
                "safetensor",
                null,
                checkpointPath.toString()).allowed();
    }

    public static String effectiveSystemPrompt(String explicitSystemPrompt, DirectSafetensorRunProfile profile) {
        if (hasText(explicitSystemPrompt)) {
            return explicitSystemPrompt;
        }
        if (Boolean.getBoolean(DISABLE_DEFAULT_RUN_SYSTEM_PROPERTY)) {
            return null;
        }
        if (profile != null && profile.config() != null && profile.config().modelType() != null && profile.config().modelType().contains("qwen")) {
            return QWEN_SAFETENSOR_RUN_SYSTEM_PROMPT;
        }
        return DEFAULT_RUN_SYSTEM_PROMPT;
    }

    public static boolean shouldForwardSystemPromptToDirectPath(String explicitSystemPrompt, DirectSafetensorRunProfile profile) {
        return hasText(explicitSystemPrompt) || (profile != null && profile.config() != null && profile.config().modelType() != null && profile.config().modelType().contains("qwen"));
    }

    public static boolean shouldUseDirectLiteRtStreamPath(String currentProvider, String localPath) {
        if (!"litert".equalsIgnoreCase(currentProvider) || !hasText(localPath)) {
            return false;
        }
        if (!Boolean.getBoolean("gollek.cli.enable_direct_litert_stream")) {
            return false;
        }
        return looksLikeLiteRtArtifactOrDirectory(localPath);
    }

    public static AlternateRuntimeSelection selectGemma3AlternateRuntime(String currentProvider, String requestedModelId,
            String localPath, boolean providerExplicit, boolean preferAlternateRuntime,
            Predicate<String> providerActive) {
        if (!allowGemma3AlternateRuntime(providerExplicit, preferAlternateRuntime)) {
            return AlternateRuntimeSelection.none();
        }
        if (!"safetensor".equalsIgnoreCase(currentProvider) || !hasText(localPath)) {
            return AlternateRuntimeSelection.none();
        }
        try {
            Optional<Path> modelDir = modelDirectory(localPath);
            if (modelDir.isEmpty() || !DirectSafetensorRunProfile.load(modelDir.get()).supportsAlternateRuntime("litert")) {
                return AlternateRuntimeSelection.none();
            }

            Optional<Path> litert = findPreferredAlternateArtifact(modelDir.get(), requestedModelId, LITERT_SUFFIXES);
            if (litert.isPresent() && providerActive.test("litert")) {
                Path selected = litert.get().toAbsolutePath().normalize();
                return new AlternateRuntimeSelection(
                        "litert",
                        selected.toString(),
                        "litert",
                        "Detected Gemma3 safetensor checkpoint; switching to LiteRT runtime artifact: " + selected,
                        false,
                        null);
            }

            Optional<Path> gguf = findPreferredAlternateArtifact(modelDir.get(), requestedModelId, GGUF_SUFFIXES);
            if (gguf.isPresent() && providerActive.test("gguf")) {
                Path selected = gguf.get().toAbsolutePath().normalize();
                return new AlternateRuntimeSelection(
                        "gguf",
                        selected.toString(),
                        "gguf",
                        "Detected Gemma3 safetensor checkpoint; switching to GGUF runtime artifact: " + selected,
                        false,
                        null);
            }

            if (litert.isPresent() || gguf.isPresent()) {
                return new AlternateRuntimeSelection(
                        null,
                        null,
                        null,
                        "Detected Gemma3 alternate runtime artifact, but provider is not currently available in this build."
                                + " Available providers are shown by: gollek modules",
                        false,
                        null);
            }
        } catch (Exception ignored) {
            // Keep original route on any failure.
        }
        return AlternateRuntimeSelection.none();
    }

    public static AlternateRuntimeSelection selectGemma4MobileQatAlternateRuntime(
            String currentProvider,
            String requestedModelId,
            String localPath,
            boolean providerExplicit,
            boolean preferAlternateRuntime,
            Predicate<String> providerActive) {
        return selectGemma4MobileQatAlternateRuntime(
                currentProvider,
                requestedModelId,
                localPath,
                providerExplicit,
                preferAlternateRuntime,
                providerActive,
                resolver::resolveLiteRtModel);
    }

    public static AlternateRuntimeSelection selectGemma4MobileQatAlternateRuntime(
            String currentProvider,
            String requestedModelId,
            String localPath,
            boolean providerExplicit,
            boolean preferAlternateRuntime,
            Predicate<String> providerActive,
            Function<String, Optional<Path>> litertResolver) {
        if (!allowGemma4MobileQatAlternateRuntime(providerExplicit, preferAlternateRuntime)) {
            return AlternateRuntimeSelection.none();
        }
        if (!"safetensor".equalsIgnoreCase(currentProvider) || !hasText(localPath)) {
            return AlternateRuntimeSelection.none();
        }
        try {
            Optional<Path> modelDir = modelDirectory(localPath);
            if (modelDir.isEmpty() || !isGemma4MobileQat(modelDir.get())) {
                return AlternateRuntimeSelection.none();
            }
            Optional<Path> cachedLiteRt = findCachedGemma4LiteRtEquivalent(requestedModelId, localPath);
            Optional<Path> localLiteRt = cachedLiteRt
                    .or(() -> resolveLiteRtEquivalent(litertResolver, requestedModelId, localPath))
                    .or(() -> findLocalGemma4LiteRtEquivalent(requestedModelId, localPath));
            if (localLiteRt.isPresent()) {
                Path selected = localLiteRt.get().toAbsolutePath().normalize();
                boolean cacheHit = cachedLiteRt.isPresent();
                if (!cacheHit) {
                    cacheGemma4LiteRtEquivalent(requestedModelId, localPath, selected);
                }
                String notice = cacheHit
                        ? null
                        : providerActive.test("litert")
                        ? "Detected Gemma 4 mobile QAT safetensor checkpoint; switching to local LiteRT-LM artifact: "
                                + selected
                        : "Detected Gemma 4 mobile QAT safetensor checkpoint and a local LiteRT-LM artifact; "
                                + "litert provider is not reported active, trying the local LiteRT-LM fallback: "
                                + selected;
                return new AlternateRuntimeSelection(
                        "litert",
                        selected.toString(),
                        "litert",
                        notice,
                        cacheHit,
                        cacheHit ? GEMMA4_MOBILE_QAT_LITERT_CACHE_KIND : null);
            }
        } catch (Exception ignored) {
            // Keep original route on any failure.
        }
        return AlternateRuntimeSelection.none();
    }

    public static AlternateRuntimeSelection selectCachedGemma4MobileQatLiteRtAlternateRuntime(
            String currentProvider,
            String requestedModelId,
            String localPath,
            boolean providerExplicit,
            boolean preferAlternateRuntime) {
        if (!allowGemma4MobileQatAlternateRuntime(providerExplicit, preferAlternateRuntime)) {
            return AlternateRuntimeSelection.none();
        }
        if (!"safetensor".equalsIgnoreCase(currentProvider) || !hasText(localPath)) {
            return AlternateRuntimeSelection.none();
        }
        try {
            Optional<Path> modelDir = modelDirectory(localPath);
            if (modelDir.isEmpty() || !isGemma4MobileQat(modelDir.get())) {
                return AlternateRuntimeSelection.none();
            }
            return findCachedGemma4LiteRtEquivalent(requestedModelId, localPath)
                    .map(path -> new AlternateRuntimeSelection(
                            "litert",
                            path.toAbsolutePath().normalize().toString(),
                            "litert",
                            null,
                            true,
                            GEMMA4_MOBILE_QAT_LITERT_CACHE_KIND))
                    .orElseGet(AlternateRuntimeSelection::none);
        } catch (Exception ignored) {
            return AlternateRuntimeSelection.none();
        }
    }

    public static AlternateRuntimeSelection selectGemma4TextAlternateRuntime(
            String currentProvider,
            String requestedModelId,
            String localPath,
            boolean providerExplicit,
            boolean preferAlternateRuntime,
            Predicate<String> providerActive) {
        return selectGemma4TextAlternateRuntime(
                currentProvider,
                requestedModelId,
                localPath,
                providerExplicit,
                preferAlternateRuntime,
                providerActive,
                DirectSafetensorRoutePolicy::findLocalGemma4GgufEquivalent);
    }

    public static AlternateRuntimeSelection selectGemma4TextAlternateRuntime(
            String currentProvider,
            String requestedModelId,
            String localPath,
            boolean providerExplicit,
            boolean preferAlternateRuntime,
            Predicate<String> providerActive,
            BiFunction<String, String, Optional<Path>> ggufResolver) {
        if (!allowGemma4TextAlternateRuntime(providerExplicit, preferAlternateRuntime)) {
            return AlternateRuntimeSelection.none();
        }
        if (!"safetensor".equalsIgnoreCase(currentProvider) || !hasText(localPath)) {
            return AlternateRuntimeSelection.none();
        }
        try {
            Optional<Path> modelDir = modelDirectory(localPath);
            if (modelDir.isEmpty()) {
                return AlternateRuntimeSelection.none();
            }
            DirectSafetensorRunProfile profile = DirectSafetensorRunProfile.load(modelDir.get());
            if (!profile.nativeBf16Matvec()
                    || isGemma4Unified(profile)
                    || isGemma4MobileQat(modelDir.get())) {
                return AlternateRuntimeSelection.none();
            }

            Optional<Path> cachedGguf = findCachedGemma4TextGgufEquivalent(requestedModelId, localPath);
            Optional<Path> localGguf = cachedGguf
                    .or(() -> findPreferredAlternateArtifact(modelDir.get(), requestedModelId, GGUF_SUFFIXES))
                    .or(() -> resolveGgufEquivalent(ggufResolver, requestedModelId, localPath));
            if (localGguf.isPresent() && providerActive.test("gguf")) {
                Path selected = localGguf.get().toAbsolutePath().normalize();
                boolean cacheHit = cachedGguf.isPresent();
                if (!cacheHit) {
                    cacheGemma4TextGgufEquivalent(requestedModelId, localPath, selected);
                }
                return new AlternateRuntimeSelection(
                        "gguf",
                        selected.toString(),
                        "gguf",
                        cacheHit
                                ? null
                                : "Detected Gemma 4 safetensor checkpoint with local GGUF runtime artifact; "
                                        + "switching to GGUF for faster production inference: " + selected,
                        cacheHit,
                        cacheHit ? GEMMA4_TEXT_GGUF_CACHE_KIND : null);
            }
            if (localGguf.isPresent()) {
                cacheGemma4TextGgufEquivalent(requestedModelId, localPath, localGguf.get());
                return new AlternateRuntimeSelection(
                        null,
                        null,
                        null,
                        "Detected Gemma 4 GGUF runtime artifact, but the gguf provider is not currently available "
                                + "in this build. Available providers are shown by: gollek modules",
                        false,
                        null);
            }
        } catch (Exception ignored) {
            // Keep original route on any failure.
        }
        return AlternateRuntimeSelection.none();
    }

    public static AlternateRuntimeSelection selectCommunityTextGgufAlternateRuntime(
            String currentProvider,
            String requestedModelId,
            String localPath,
            boolean providerExplicit,
            boolean preferAlternateRuntime,
            Predicate<String> providerActive) {
        return selectCommunityTextGgufAlternateRuntime(
                currentProvider,
                requestedModelId,
                localPath,
                providerExplicit,
                preferAlternateRuntime,
                providerActive,
                DirectSafetensorRoutePolicy::findLocalCommunityTextGgufEquivalent);
    }

    public static AlternateRuntimeSelection selectCommunityTextGgufAlternateRuntime(
            String currentProvider,
            String requestedModelId,
            String localPath,
            boolean providerExplicit,
            boolean preferAlternateRuntime,
            Predicate<String> providerActive,
            BiFunction<String, String, Optional<Path>> ggufResolver) {
        if (!allowCommunityTextGgufAlternateRuntime(providerExplicit, preferAlternateRuntime)) {
            return AlternateRuntimeSelection.none();
        }
        if (!"safetensor".equalsIgnoreCase(currentProvider) || !hasText(localPath)) {
            return AlternateRuntimeSelection.none();
        }
        try {
            Optional<Path> modelDir = modelDirectory(localPath);
            if (modelDir.isEmpty()) {
                return AlternateRuntimeSelection.none();
            }
            DirectSafetensorRunProfile profile = DirectSafetensorRunProfile.load(modelDir.get());
            Optional<CommunityTextGgufFamily> family = communityTextGgufFamily(profile);
            if (family.isEmpty()) {
                return AlternateRuntimeSelection.none();
            }

            CommunityTextGgufFamily selectedFamily = family.get();
            Optional<Path> cachedGguf =
                    findCachedCommunityTextGgufEquivalent(selectedFamily, requestedModelId, localPath);
            Optional<Path> localGguf = cachedGguf
                    .or(() -> findPreferredAlternateArtifact(modelDir.get(), requestedModelId, GGUF_SUFFIXES))
                    .or(() -> resolveGgufEquivalent(ggufResolver, requestedModelId, localPath));
            if (localGguf.isPresent() && providerActive.test("gguf")) {
                Path selected = localGguf.get().toAbsolutePath().normalize();
                boolean cacheHit = cachedGguf.isPresent();
                if (!cacheHit) {
                    cacheCommunityTextGgufEquivalent(selectedFamily, requestedModelId, localPath, selected);
                }
                return new AlternateRuntimeSelection(
                        "gguf",
                        selected.toString(),
                        "gguf",
                        cacheHit
                                ? null
                                : "Detected " + selectedFamily.label()
                                        + " safetensor checkpoint with local GGUF runtime artifact; "
                                        + "switching to GGUF for faster production inference: " + selected,
                        cacheHit,
                        cacheHit ? COMMUNITY_TEXT_GGUF_CACHE_KIND : null);
            }
            if (localGguf.isPresent()) {
                cacheCommunityTextGgufEquivalent(selectedFamily, requestedModelId, localPath, localGguf.get());
                return new AlternateRuntimeSelection(
                        null,
                        null,
                        null,
                        "Detected " + selectedFamily.label()
                                + " GGUF runtime artifact, but the gguf provider is not currently available "
                                + "in this build. Available providers are shown by: gollek modules",
                        false,
                        null);
            }
        } catch (Exception ignored) {
            // Keep original route on any failure.
        }
        return AlternateRuntimeSelection.none();
    }

    public static RouteValidation validateGemma3ExecutionRoute(String provider, String requestedModelId, String localPath) {
        if (!"safetensor".equalsIgnoreCase(provider) || !hasText(localPath)) {
            return RouteValidation.pass();
        }
        try {
            Optional<Path> modelDir = modelDirectory(localPath);
            if (modelDir.isEmpty() || !DirectSafetensorRunProfile.load(modelDir.get()).supportsAlternateRuntime("litert")) {
                return RouteValidation.pass();
            }
            String hint = normalizeArtifactHint(requestedModelId);
            if (hint.contains("functiongemma")) {
                return RouteValidation.pass();
            }
            Optional<Path> litert = findPreferredAlternateArtifact(modelDir.get(), requestedModelId, LITERT_SUFFIXES);
            Optional<Path> gguf = findPreferredAlternateArtifact(modelDir.get(), requestedModelId, GGUF_SUFFIXES);
            if (litert.isPresent() || gguf.isPresent()) {
                return RouteValidation.pass();
            }

            return RouteValidation.invalid(List.of(
                    "Error: Gemma3 safetensor direct path is disabled for quality/safety in this build.",
                    "No compatible local runtime artifact was found for this checkpoint.",
                    "Try:",
                    "  gollek pull gguf:google/functiongemma-270m-it",
                    "  gollek run --provider gguf --model google/functiongemma-270m-it --prompt \"who are you\"",
                    "If gguf provider is unavailable, rebuild with gguf runtime/plugin enabled."));
        } catch (Exception ignored) {
            return RouteValidation.pass();
        }
    }

    public static RouteValidation validateGemma4UnifiedExecutionRoute(
            String provider,
            String requestedModelId,
            String localPath) {
        if (!"safetensor".equalsIgnoreCase(provider) || !hasText(localPath)) {
            return RouteValidation.pass();
        }
        try {
            Optional<Path> modelDir = modelDirectory(localPath);
            if (modelDir.isEmpty()) {
                return RouteValidation.pass();
            }
            DirectSafetensorRunProfile profile = DirectSafetensorRunProfile.load(modelDir.get());
            Gemma4SafetensorExecutionProfile executionProfile =
                    Gemma4SafetensorExecutionProfile.from(profile, isGemma4MobileQat(modelDir.get()));
            if (executionProfile.mobileQatCheckpoint()) {
                return validateGemma4MobileQatExecutionRoute(requestedModelId, modelDir.get(), executionProfile);
            }
            if (!executionProfile.unifiedMultimodal()) {
                return RouteValidation.pass();
            }
            if (executionProfile.packedMoeCheckpoint()) {
                return validateGemma4PackedMoeExecutionRoute(
                        requestedModelId,
                        modelDir.get(),
                        profile.config(),
                        executionProfile);
            }
            if (executionProfile.guardedTextDecoderReady()) {
                Gemma4UnifiedSafetensorPreflight.Result preflight =
                        Gemma4UnifiedSafetensorPreflight.validate(
                                modelDir.get(),
                                profile.config(),
                                hasText(requestedModelId)
                                        ? requestedModelId
                                        : modelDir.get().getFileName().toString());
                if (!preflight.allowed()) {
                    return RouteValidation.invalid(
                            preflight.messages(),
                            mergeDetails(executionProfile.diagnosticDetails(), preflight.diagnosticDetails()));
                }
                return RouteValidation.pass();
            }
            String modelLabel = hasText(requestedModelId)
                    ? requestedModelId
                    : modelDir.get().getFileName().toString();
            return RouteValidation.invalid(List.of(
                    "Error: Gemma 4 unified safetensor runtime is not enabled in this build.",
                    "Checkpoint " + modelLabel
                            + " is recognized as gemma4_unified, but the local direct safetensor path only covers the guarded dense Gemma 4 text adapter.",
                    "Use a runtime plugin with Gemma 4 unified multimodal embedder support, convert to a supported artifact, or detach Gemma 4 unified from production direct bundles."),
                    executionProfile.diagnosticDetails());
        } catch (Exception ignored) {
            return RouteValidation.pass();
        }
    }

    public static RouteValidation validateGemma4UnifiedMultimodalExecutionRoute(
            String provider,
            String requestedModelId,
            String localPath,
            boolean hasVisionInput,
            boolean ocrMode) {
        if (!"safetensor".equalsIgnoreCase(provider) || !hasText(localPath) || (!hasVisionInput && !ocrMode)) {
            return RouteValidation.pass();
        }
        try {
            Optional<Path> modelDir = modelDirectory(localPath);
            if (modelDir.isEmpty()) {
                return RouteValidation.pass();
            }
            DirectSafetensorRunProfile profile = DirectSafetensorRunProfile.load(modelDir.get());
            Gemma4SafetensorExecutionProfile executionProfile =
                    Gemma4SafetensorExecutionProfile.from(profile, isGemma4MobileQat(modelDir.get()));
            if (executionProfile.mobileQatCheckpoint()) {
                return validateGemma4MobileQatExecutionRoute(requestedModelId, modelDir.get(), executionProfile);
            }
            if (!executionProfile.unifiedMultimodal()) {
                return RouteValidation.pass();
            }
            if (executionProfile.packedMoeCheckpoint()) {
                return validateGemma4PackedMoeExecutionRoute(
                        requestedModelId,
                        modelDir.get(),
                        profile.config(),
                        executionProfile);
            }
            String modelLabel = hasText(requestedModelId)
                    ? requestedModelId
                    : modelDir.get().getFileName().toString();
            String requestMode = ocrMode ? "--ocr" : "--image";
            Gemma4UnifiedSafetensorPreflight.Inspection inspection =
                    Gemma4UnifiedSafetensorPreflight.inspect(
                            modelDir.get(),
                            profile.config(),
                            modelLabel);
            Gemma4UnifiedSafetensorPreflight.ProjectorSummary projectors = inspection.projectors();
            return RouteValidation.invalid(List.of(
                    "Error: Gemma 4 unified multimodal safetensor runtime is not enabled in this build.",
                    "Checkpoint " + modelLabel
                            + " is a gemma4_unified checkpoint and the request includes " + requestMode
                            + " input, but the local direct safetensor path only runs the guarded dense text decoder.",
                    "Header preflight: " + projectors.display() + "; no 12B weight payload was loaded.",
                    "Implement or enable the Gemma 4 vision/audio/video embedder and projector path before routing multimodal requests to the safetensor runner."),
                    mergeDetails(executionProfile.diagnosticDetails(), Map.of(
                            ProblemDetail.HEADER_PREFLIGHT, projectors.display(),
                            ProblemDetail.HEADER_INSPECTION, inspection.header().diagnosticDetails(),
                            ProblemDetail.TENSOR_INVENTORY, inspection.inventory().diagnosticDetails(),
                            ProblemDetail.COMPONENT_READINESS, inspection.readiness().diagnosticDetails(),
                            ProblemDetail.DETECTED_PROJECTORS, projectors.detectedProjectors(),
                            ProblemDetail.DETECTED_PACKED_MOE, projectors.detectedPackedMoe())));
        } catch (Exception ignored) {
            return RouteValidation.pass();
        }
    }

    private static RouteValidation validateGemma4PackedMoeExecutionRoute(
            String requestedModelId,
            Path modelDir,
            ModelConfig config,
            Gemma4SafetensorExecutionProfile executionProfile) {
        String modelLabel = hasText(requestedModelId)
                ? requestedModelId
                : modelDir.getFileName().toString();
        Map<String, Object> details = new LinkedHashMap<>(executionProfile.diagnosticDetails());
        Gemma4UnifiedSafetensorPreflight.Inspection inspection =
                Gemma4UnifiedSafetensorPreflight.inspect(modelDir, config, modelLabel);
        details = mergeDetails(details, Map.of(
                ProblemDetail.HEADER_PREFLIGHT, inspection.projectors().display(),
                ProblemDetail.HEADER_INSPECTION, inspection.header().diagnosticDetails(),
                ProblemDetail.TENSOR_INVENTORY, inspection.inventory().diagnosticDetails(),
                ProblemDetail.COMPONENT_READINESS, inspection.readiness().diagnosticDetails(),
                ProblemDetail.DETECTED_PROJECTORS, inspection.projectors().detectedProjectors(),
                ProblemDetail.DETECTED_PACKED_MOE, inspection.projectors().detectedPackedMoe()));
        return RouteValidation.invalid(List.of(
                "Error: Gemma 4 packed MoE safetensor runtime is not enabled in this build.",
                "Checkpoint " + modelLabel
                        + " declares Gemma 4 packed expert routing, but the local direct safetensor path only covers the guarded dense Gemma 4 text adapter.",
                "Header preflight: " + inspection.projectors().display() + "; no 12B weight payload was loaded.",
                "Use a runtime plugin with Gemma 4 packed-expert routing support, convert to a supported artifact, or keep packed MoE checkpoints out of direct safetensor production bundles."),
                details);
    }

    private static RouteValidation validateGemma4MobileQatExecutionRoute(
            String requestedModelId,
            Path modelDir,
            Gemma4SafetensorExecutionProfile executionProfile) {
        try {
            if (executionProfile == null || !executionProfile.mobileQatCheckpoint()) {
                return RouteValidation.pass();
            }
            String modelLabel = hasText(requestedModelId)
                    ? requestedModelId
                    : modelDir.getFileName().toString();
            return RouteValidation.invalid(List.of(
                    "Error: Gemma 4 mobile QAT safetensor runtime is not enabled in this build.",
                    "Checkpoint " + modelLabel
                            + " uses Google Gemma quantization with mobile audio/vision/text towers, but the local direct safetensor path does not have the mobile QAT loader yet.",
                    "Use a LiteRT-LM/mobile runtime artifact, a supported quantized runner, or install a build with Gemma 4 mobile QAT loader support."),
                    executionProfile.diagnosticDetails());
        } catch (Exception ignored) {
            return RouteValidation.pass();
        }
    }

    private static boolean allowGemma3AlternateRuntime(boolean providerExplicit, boolean preferAlternateRuntime) {
        if (preferAlternateRuntime) {
            return true;
        }
        if (providerExplicit) {
            return false;
        }
        return Boolean.getBoolean(ENABLE_GEMMA3_ALTERNATE_RUNTIME_PROPERTY);
    }

    private static boolean allowGemma4MobileQatAlternateRuntime(boolean providerExplicit, boolean preferAlternateRuntime) {
        if (preferAlternateRuntime) {
            return true;
        }
        return !providerExplicit;
    }

    private static boolean allowGemma4TextAlternateRuntime(boolean providerExplicit, boolean preferAlternateRuntime) {
        if (preferAlternateRuntime) {
            return true;
        }
        return !providerExplicit;
    }

    private static boolean allowCommunityTextGgufAlternateRuntime(
            boolean providerExplicit,
            boolean preferAlternateRuntime) {
        if (preferAlternateRuntime) {
            return true;
        }
        return !providerExplicit;
    }

    private static Optional<Path> resolveLiteRtEquivalent(
            Function<String, Optional<Path>> resolver,
            String requestedModelId,
            String localPath) {
        if (resolver == null) {
            return Optional.empty();
        }
        for (String candidate : List.of(requestedModelId, localPath)) {
            if (!hasText(candidate)) {
                continue;
            }
            Optional<Path> resolved = resolver.apply(candidate);
            if (resolved != null && resolved.isPresent()) {
                return resolved;
            }
        }
        return Optional.empty();
    }

    private static Optional<Path> resolveGgufEquivalent(
            BiFunction<String, String, Optional<Path>> resolver,
            String requestedModelId,
            String localPath) {
        if (resolver == null) {
            return Optional.empty();
        }
        Optional<Path> resolved = resolver.apply(requestedModelId, localPath);
        return resolved == null ? Optional.empty() : resolved;
    }

    private static Optional<Path> findLocalGemma4LiteRtEquivalent(String requestedModelId, String localPath) {
        String requestedKey = gemma4EquivalentKey(requestedModelId + " " + localPath);
        if (requestedKey.isBlank()) {
            return Optional.empty();
        }
        Path root = Path.of(System.getProperty("user.home"), ".gollek", "models");
        if (!Files.isDirectory(root)) {
            return Optional.empty();
        }
        try (var stream = Files.walk(root, 8)) {
            return stream
                    .filter(Files::isRegularFile)
                    .filter(DirectSafetensorRoutePolicy::isLiteRtLmFile)
                    .filter(candidate -> gemma4LiteRtMatches(requestedKey, candidate))
                    .sorted((left, right) -> Integer.compare(
                            gemma4LiteRtPreference(left),
                            gemma4LiteRtPreference(right)))
                    .findFirst()
                    .map(candidate -> candidate.toAbsolutePath().normalize());
        } catch (Exception ignored) {
            return Optional.empty();
        }
    }

    private static Optional<Path> findLocalGemma4GgufEquivalent(String requestedModelId, String localPath) {
        Set<String> requestedKeys = gemma4GgufTargetKeys(requestedModelId, localPath);
        if (requestedKeys.isEmpty()) {
            return Optional.empty();
        }
        return resolver.localModels()
                .filter(entry -> entry != null && "gguf".equalsIgnoreCase(entry.format()))
                .filter(entry -> entry.path() != null && !entry.path().toString().isBlank())
                .map(entry -> new Gemma4GgufCandidate(entry, entry.path()))
                .filter(candidate -> Files.isRegularFile(candidate.path()))
                .filter(candidate -> gemma4GgufMatches(requestedKeys, candidate.entry()))
                .min(Comparator.comparingInt(DirectSafetensorRoutePolicy::gemma4GgufPreference))
                .map(Gemma4GgufCandidate::path)
                .map(path -> path.toAbsolutePath().normalize());
    }

    private record Gemma4GgufCandidate(LocalModelResolver.LocalModelCandidate entry, Path path) {
    }

    private record CommunityTextGgufFamily(String id, String label) {
    }

    private record CommunityTextGgufCandidate(LocalModelResolver.LocalModelCandidate entry, Path path) {
    }

    private static Optional<CommunityTextGgufFamily> communityTextGgufFamily(
            DirectSafetensorRunProfile profile) {
        if (profile == null || profile.config() == null) {
            return Optional.empty();
        }
        String modelType = communityClassifierToken(profile.config().modelType());
        String architecture = communityClassifierToken(profile.config().primaryArchitecture());
        if (isExcludedCommunityTextGgufFamily(modelType, architecture)) {
            return Optional.empty();
        }
        Optional<CommunityTextGgufFamily> byModelType = communityTextGgufFamilyFromToken(modelType);
        if (byModelType.isPresent()) {
            return byModelType;
        }
        return communityTextGgufFamilyFromToken(architecture);
    }

    private static Optional<CommunityTextGgufFamily> communityTextGgufFamilyFromToken(String token) {
        if (token == null || token.isBlank()) {
            return Optional.empty();
        }
        if (token.startsWith("qwen")) {
            return Optional.of(new CommunityTextGgufFamily("qwen", "Qwen"));
        }
        if (token.equals("llama")
                || token.startsWith("llama")
                || token.equals("code_llama")
                || token.startsWith("codellama")) {
            return Optional.of(new CommunityTextGgufFamily("llama", "Llama"));
        }
        if (token.startsWith("phi")) {
            return Optional.of(new CommunityTextGgufFamily("phi", "Phi"));
        }
        if (token.startsWith("mistral")
                || token.startsWith("mixtral")
                || token.startsWith("ministral")) {
            return Optional.of(new CommunityTextGgufFamily("mistral", "Mistral"));
        }
        if (token.startsWith("falcon")) {
            return Optional.of(new CommunityTextGgufFamily("falcon", "Falcon"));
        }
        if (token.equals("gemma") || token.equals("gemma2")) {
            return Optional.of(new CommunityTextGgufFamily("gemma", "Gemma"));
        }
        if (token.startsWith("deepseek")) {
            return Optional.of(new CommunityTextGgufFamily("deepseek", "DeepSeek"));
        }
        if (token.equals("yi") || token.startsWith("yi_")) {
            return Optional.of(new CommunityTextGgufFamily("yi", "Yi"));
        }
        if (token.startsWith("cohere")) {
            return Optional.of(new CommunityTextGgufFamily("cohere", "Cohere"));
        }
        if (token.startsWith("granite")) {
            return Optional.of(new CommunityTextGgufFamily("granite", "Granite"));
        }
        if (token.startsWith("olmo")) {
            return Optional.of(new CommunityTextGgufFamily("olmo", "OLMo"));
        }
        if (token.startsWith("mpt")) {
            return Optional.of(new CommunityTextGgufFamily("mpt", "MPT"));
        }
        if (token.startsWith("bloom")) {
            return Optional.of(new CommunityTextGgufFamily("bloom", "BLOOM"));
        }
        if (token.startsWith("gpt_neox")
                || token.startsWith("gptj")
                || token.startsWith("gpt2")) {
            return Optional.of(new CommunityTextGgufFamily("gpt", "GPT-style"));
        }
        if (token.startsWith("starcoder")) {
            return Optional.of(new CommunityTextGgufFamily("starcoder", "StarCoder"));
        }
        if (token.startsWith("stablelm")) {
            return Optional.of(new CommunityTextGgufFamily("stablelm", "StableLM"));
        }
        if (token.startsWith("smollm")) {
            return Optional.of(new CommunityTextGgufFamily("smollm", "SmolLM"));
        }
        return Optional.empty();
    }

    private static boolean isExcludedCommunityTextGgufFamily(String modelType, String architecture) {
        String combined = (modelType == null ? "" : modelType)
                + " "
                + (architecture == null ? "" : architecture);
        if (combined.isBlank()) {
            return true;
        }
        if (modelType.startsWith("gemma3") || modelType.startsWith("gemma4")) {
            return true;
        }
        if (modelType.contains("_vl")
                || modelType.endsWith("vl")
                || modelType.contains("_omni")
                || modelType.contains("_audio")
                || architecture.contains("vlfor")
                || architecture.contains("vision")
                || architecture.contains("image")
                || architecture.contains("audio")) {
            return true;
        }
        return containsAny(
                combined,
                "bert",
                "flava",
                "clip",
                "llava",
                "paligemma",
                "mllama",
                "pixtral",
                "chameleon",
                "kosmos",
                "blip",
                "whisper",
                "speech",
                "ocr",
                "video",
                "multimodal",
                "embedding",
                "maskedlm",
                "sequenceclassification",
                "tokenclassification",
                "mamba");
    }

    private static String communityClassifierToken(String value) {
        if (value == null) {
            return "";
        }
        return value.trim()
                .toLowerCase(Locale.ROOT)
                .replace('-', '_')
                .replace('.', '_');
    }

    private static Optional<Path> findLocalCommunityTextGgufEquivalent(String requestedModelId, String localPath) {
        Set<String> requestedKeys = communityTextGgufTargetKeys(requestedModelId, localPath);
        if (requestedKeys.isEmpty()) {
            return Optional.empty();
        }
        return resolver.localModels()
                .filter(entry -> entry != null && "gguf".equalsIgnoreCase(entry.format()))
                .filter(entry -> entry.path() != null && !entry.path().toString().isBlank())
                .map(entry -> new CommunityTextGgufCandidate(entry, entry.path()))
                .filter(candidate -> Files.isRegularFile(candidate.path()))
                .filter(candidate -> communityTextGgufMatches(requestedKeys, candidate.entry()))
                .min(Comparator.comparingInt(DirectSafetensorRoutePolicy::communityTextGgufPreference))
                .map(CommunityTextGgufCandidate::path)
                .map(path -> path.toAbsolutePath().normalize());
    }

    private static Set<String> communityTextGgufTargetKeys(String requestedModelId, String localPath) {
        Set<String> keys = new LinkedHashSet<>();
        if (!looksLikeShortId(requestedModelId)) {
            addCommunityTextGgufKey(keys, requestedModelId);
        }
        try {
            Path path = Path.of(localPath);
            addCommunityTextGgufKey(keys, fileName(path));
            Path dir = Files.isDirectory(path) ? path : path.getParent();
            addCommunityTextGgufKey(keys, fileName(dir));
            if (dir != null) {
                addCommunityTextGgufKey(keys, fileName(dir.getParent()));
            }
        } catch (Exception ignored) {
            addCommunityTextGgufKey(keys, localPath);
        }
        return keys;
    }

    private static boolean communityTextGgufMatches(Set<String> requestedKeys, LocalModelResolver.LocalModelCandidate entry) {
        Set<String> candidateKeys = new LinkedHashSet<>();
        addCommunityTextGgufKey(candidateKeys, entry.id());
        addCommunityTextGgufKey(candidateKeys, entry.name());
        addCommunityTextGgufKey(candidateKeys, fileName(entry.path()));
        addCommunityTextGgufKey(candidateKeys, entry.path().toString());
        try {
            addCommunityTextGgufKey(candidateKeys, fileName(entry.path()));
        } catch (Exception ignored) {
            addCommunityTextGgufKey(candidateKeys, entry.path().toString());
        }
        for (String requested : requestedKeys) {
            for (String candidate : candidateKeys) {
                if (candidate.contains(requested) || requested.contains(candidate)) {
                    return true;
                }
            }
        }
        return false;
    }

    private static boolean communityTextGgufPathMatches(Set<String> requestedKeys, Path path) {
        Set<String> candidateKeys = new LinkedHashSet<>();
        addCommunityTextGgufKey(candidateKeys, fileName(path));
        try {
            addCommunityTextGgufKey(candidateKeys, path.toAbsolutePath().normalize().toString());
        } catch (Exception ignored) {
            addCommunityTextGgufKey(candidateKeys, path.toString());
        }
        for (String requested : requestedKeys) {
            for (String candidate : candidateKeys) {
                if (candidate.contains(requested) || requested.contains(candidate)) {
                    return true;
                }
            }
        }
        return false;
    }

    private static int communityTextGgufPreference(CommunityTextGgufCandidate candidate) {
        LocalModelResolver.LocalModelCandidate entry = candidate.entry();
        String text = ((entry.id() == null ? "" : entry.id())
                + " "
                + (entry.name() == null ? "" : entry.name())
                + " "
                + candidate.path()).toLowerCase(Locale.ROOT);
        int score = 100;
        if (text.contains("q4_k_m") || text.contains("q4/k/m")) {
            score -= 40;
        } else if (text.contains("q4")) {
            score -= 30;
        } else if (text.contains("q5")) {
            score -= 25;
        } else if (text.contains("q8")) {
            score -= 15;
        }
        if (text.contains("i2") || text.contains("iq2")) {
            score += 30;
        }
        return score;
    }

    private static void addCommunityTextGgufKey(Set<String> keys, String value) {
        String key = communityTextGgufEquivalentKey(value);
        if (key.length() >= 6) {
            keys.add(key);
        }
    }

    private static String communityTextGgufEquivalentKey(String value) {
        if (value == null || value.isBlank()) {
            return "";
        }
        String normalized = value.toLowerCase(Locale.ROOT)
                .replace("__", "/")
                .replace('\\', '/')
                .replaceAll("\\.(gguf|safetensors?|bin|pt|pth|onnx)$", " ");
        StringBuilder key = new StringBuilder();
        for (String token : normalized.split("[^a-z0-9]+")) {
            if (isCommunityTextGgufStopToken(token)) {
                continue;
            }
            key.append(token);
        }
        return key.toString();
    }

    private static boolean isCommunityTextGgufStopToken(String token) {
        if (token == null || token.isBlank()) {
            return true;
        }
        String normalized = token.toLowerCase(Locale.ROOT);
        return switch (normalized) {
            case "gguf", "ggml", "safetensor", "safetensors", "safe", "tensors",
                    "transformers", "model", "models", "blob", "blobs", "hf", "hub",
                    "local", "quant", "quantized", "q", "k", "m", "km",
                    "thebloke", "unsloth", "community", "mlx", "litert", "task",
                    "google", "microsoft", "meta", "facebook", "tiiuae", "mistralai",
                    "01ai", "allenai" -> true;
            default -> normalized.matches("q\\d+[a-z]*")
                    || normalized.matches("iq\\d+[a-z]*")
                    || normalized.matches("f\\d+")
                    || normalized.matches("fp\\d+")
                    || normalized.matches("bf\\d+");
        };
    }

    private static Set<String> gemma4GgufTargetKeys(String requestedModelId, String localPath) {
        Set<String> keys = new LinkedHashSet<>();
        if (!looksLikeShortId(requestedModelId)) {
            addGemma4GgufKey(keys, requestedModelId);
        }
        try {
            Path path = Path.of(localPath);
            addGemma4GgufKey(keys, fileName(path));
            Path dir = Files.isDirectory(path) ? path : path.getParent();
            addGemma4GgufKey(keys, fileName(dir));
            if (dir != null) {
                addGemma4GgufKey(keys, fileName(dir.getParent()));
            }
        } catch (Exception ignored) {
            addGemma4GgufKey(keys, localPath);
        }
        return keys;
    }

    private static boolean gemma4GgufMatches(Set<String> requestedKeys, LocalModelResolver.LocalModelCandidate entry) {
        Set<String> candidateKeys = new LinkedHashSet<>();
        addGemma4GgufKey(candidateKeys, entry.id());
        addGemma4GgufKey(candidateKeys, entry.name());
        addGemma4GgufKey(candidateKeys, fileName(entry.path()));
        addGemma4GgufKey(candidateKeys, entry.path().toString());
        try {
            addGemma4GgufKey(candidateKeys, fileName(entry.path()));
        } catch (Exception ignored) {
            addGemma4GgufKey(candidateKeys, entry.path().toString());
        }
        for (String requested : requestedKeys) {
            for (String candidate : candidateKeys) {
                if (candidate.contains(requested) || requested.contains(candidate)) {
                    return true;
                }
            }
        }
        return false;
    }

    private static boolean gemma4GgufPathMatches(Set<String> requestedKeys, Path path) {
        Set<String> candidateKeys = new LinkedHashSet<>();
        addGemma4GgufKey(candidateKeys, fileName(path));
        try {
            addGemma4GgufKey(candidateKeys, path.toAbsolutePath().normalize().toString());
        } catch (Exception ignored) {
            addGemma4GgufKey(candidateKeys, path.toString());
        }
        for (String requested : requestedKeys) {
            for (String candidate : candidateKeys) {
                if (candidate.contains(requested) || requested.contains(candidate)) {
                    return true;
                }
            }
        }
        return false;
    }

    private static int gemma4GgufPreference(Gemma4GgufCandidate candidate) {
        LocalModelResolver.LocalModelCandidate entry = candidate.entry();
        String text = ((entry.id() == null ? "" : entry.id())
                + " "
                + (entry.name() == null ? "" : entry.name())
                + " "
                + candidate.path()).toLowerCase(Locale.ROOT);
        int score = 100;
        if (text.contains("q4_k_m") || text.contains("q4/k/m")) {
            score -= 40;
        } else if (text.contains("q4")) {
            score -= 30;
        } else if (text.contains("q5")) {
            score -= 25;
        } else if (text.contains("q8")) {
            score -= 15;
        }
        if (text.contains("i2") || text.contains("iq2")) {
            score += 30;
        }
        return score;
    }

    private static void addGemma4GgufKey(Set<String> keys, String value) {
        String key = gemma4GgufEquivalentKey(value);
        if (!key.isBlank() && key.contains("gemma4")) {
            keys.add(key);
        }
    }

    private static String gemma4GgufEquivalentKey(String value) {
        if (value == null || value.isBlank()) {
            return "";
        }
        String normalized = value.toLowerCase(Locale.ROOT)
                .replace("__", "/")
                .replace('\\', '/')
                .replaceAll("\\.(gguf|safetensors?|bin)$", " ");
        StringBuilder key = new StringBuilder();
        for (String token : normalized.split("[^a-z0-9]+")) {
            if (isGemma4GgufStopToken(token)) {
                continue;
            }
            key.append(token);
        }
        return key.toString();
    }

    private static boolean isGemma4GgufStopToken(String token) {
        if (token == null || token.isBlank()) {
            return true;
        }
        String normalized = token.toLowerCase(Locale.ROOT);
        return switch (normalized) {
            case "google", "gguf", "safetensor", "safetensors", "transformers",
                    "model", "models", "blob", "blobs", "q", "k", "m", "km" -> true;
            default -> normalized.matches("q\\d+[a-z]*")
                    || normalized.matches("iq\\d+[a-z]*");
        };
    }

    private static String fileName(Path path) {
        return path == null || path.getFileName() == null ? null : path.getFileName().toString();
    }

    private static boolean looksLikeShortId(String value) {
        if (value == null) {
            return false;
        }
        String normalized = value.trim();
        return normalized.length() >= 4
                && normalized.length() <= 12
                && normalized.chars().allMatch(ch ->
                (ch >= '0' && ch <= '9')
                        || (ch >= 'a' && ch <= 'f')
                        || (ch >= 'A' && ch <= 'F'));
    }

    private static Optional<Path> findCachedGemma4LiteRtEquivalent(String requestedModelId, String localPath) {
        String cacheKey = gemma4LiteRtCacheKey(requestedModelId, localPath);
        String requestedKey = gemma4EquivalentKey(requestedModelId + " " + localPath);
        if (requestedKey.isBlank()) {
            return Optional.empty();
        }
        return RouteArtifactCache.find(
                GEMMA4_MOBILE_QAT_LITERT_CACHE_DIR_PROPERTY,
                GEMMA4_MOBILE_QAT_LITERT_CACHE_ENABLED_PROPERTY,
                GEMMA4_MOBILE_QAT_LITERT_CACHE_FILE,
                cacheKey,
                cached -> Files.isRegularFile(cached)
                        && isLiteRtLmFile(cached)
                        && gemma4LiteRtMatches(requestedKey, cached));
    }

    private static void cacheGemma4LiteRtEquivalent(String requestedModelId, String localPath, Path selected) {
        RouteArtifactCache.put(
                GEMMA4_MOBILE_QAT_LITERT_CACHE_DIR_PROPERTY,
                GEMMA4_MOBILE_QAT_LITERT_CACHE_ENABLED_PROPERTY,
                GEMMA4_MOBILE_QAT_LITERT_CACHE_FILE,
                gemma4LiteRtCacheKey(requestedModelId, localPath),
                selected,
                DirectSafetensorRoutePolicy::isLiteRtLmFile);
    }

    private static String gemma4LiteRtCacheKey(String requestedModelId, String localPath) {
        String raw = (requestedModelId == null ? "" : requestedModelId)
                + "\n"
                + normalizedPathForCache(localPath)
                + "\n"
                + gemma4EquivalentKey(requestedModelId + " " + localPath);
        return sha256(raw);
    }

    private static Optional<Path> findCachedGemma4TextGgufEquivalent(String requestedModelId, String localPath) {
        Set<String> requestedKeys = gemma4GgufTargetKeys(requestedModelId, localPath);
        if (requestedKeys.isEmpty()) {
            return Optional.empty();
        }
        return RouteArtifactCache.find(
                GEMMA4_TEXT_GGUF_CACHE_DIR_PROPERTY,
                GEMMA4_TEXT_GGUF_CACHE_ENABLED_PROPERTY,
                GEMMA4_TEXT_GGUF_CACHE_FILE,
                nativeBf16MatvecGgufCacheKey(requestedModelId, localPath, requestedKeys),
                cached -> Files.isRegularFile(cached)
                        && hasAnySuffix(cached, GGUF_SUFFIXES)
                        && gemma4GgufPathMatches(requestedKeys, cached));
    }

    private static void cacheGemma4TextGgufEquivalent(String requestedModelId, String localPath, Path selected) {
        Set<String> requestedKeys = gemma4GgufTargetKeys(requestedModelId, localPath);
        if (requestedKeys.isEmpty()) {
            return;
        }
        RouteArtifactCache.put(
                GEMMA4_TEXT_GGUF_CACHE_DIR_PROPERTY,
                GEMMA4_TEXT_GGUF_CACHE_ENABLED_PROPERTY,
                GEMMA4_TEXT_GGUF_CACHE_FILE,
                nativeBf16MatvecGgufCacheKey(requestedModelId, localPath, requestedKeys),
                selected,
                path -> Files.isRegularFile(path)
                        && hasAnySuffix(path, GGUF_SUFFIXES)
                        && gemma4GgufPathMatches(requestedKeys, path));
    }

    private static String nativeBf16MatvecGgufCacheKey(
            String requestedModelId,
            String localPath,
            Set<String> requestedKeys) {
        String raw = (requestedModelId == null ? "" : requestedModelId)
                + "\n"
                + normalizedPathForCache(localPath)
                + "\n"
                + String.join(",", requestedKeys);
        return sha256(raw);
    }

    private static Optional<Path> findCachedCommunityTextGgufEquivalent(
            CommunityTextGgufFamily family,
            String requestedModelId,
            String localPath) {
        Set<String> requestedKeys = communityTextGgufTargetKeys(requestedModelId, localPath);
        if (family == null || requestedKeys.isEmpty()) {
            return Optional.empty();
        }
        return RouteArtifactCache.find(
                COMMUNITY_TEXT_GGUF_CACHE_DIR_PROPERTY,
                COMMUNITY_TEXT_GGUF_CACHE_ENABLED_PROPERTY,
                COMMUNITY_TEXT_GGUF_CACHE_FILE,
                communityTextGgufCacheKey(family, requestedModelId, localPath, requestedKeys),
                cached -> Files.isRegularFile(cached)
                        && hasAnySuffix(cached, GGUF_SUFFIXES)
                        && communityTextGgufPathMatches(requestedKeys, cached));
    }

    private static void cacheCommunityTextGgufEquivalent(
            CommunityTextGgufFamily family,
            String requestedModelId,
            String localPath,
            Path selected) {
        Set<String> requestedKeys = communityTextGgufTargetKeys(requestedModelId, localPath);
        if (family == null || requestedKeys.isEmpty()) {
            return;
        }
        RouteArtifactCache.put(
                COMMUNITY_TEXT_GGUF_CACHE_DIR_PROPERTY,
                COMMUNITY_TEXT_GGUF_CACHE_ENABLED_PROPERTY,
                COMMUNITY_TEXT_GGUF_CACHE_FILE,
                communityTextGgufCacheKey(family, requestedModelId, localPath, requestedKeys),
                selected,
                path -> Files.isRegularFile(path)
                        && hasAnySuffix(path, GGUF_SUFFIXES)
                        && communityTextGgufPathMatches(requestedKeys, path));
    }

    private static String communityTextGgufCacheKey(
            CommunityTextGgufFamily family,
            String requestedModelId,
            String localPath,
            Set<String> requestedKeys) {
        String raw = (family == null ? "" : family.id())
                + "\n"
                + (requestedModelId == null ? "" : requestedModelId)
                + "\n"
                + normalizedPathForCache(localPath)
                + "\n"
                + String.join(",", requestedKeys);
        return sha256(raw);
    }

    private static String normalizedPathForCache(String localPath) {
        if (!hasText(localPath)) {
            return "";
        }
        try {
            return Path.of(localPath).toAbsolutePath().normalize().toString();
        } catch (Exception ignored) {
            return localPath;
        }
    }

    private static String sha256(String value) {
        try {
            byte[] digest = MessageDigest.getInstance("SHA-256")
                    .digest(value.getBytes(StandardCharsets.UTF_8));
            StringBuilder hex = new StringBuilder(digest.length * 2);
            for (byte b : digest) {
                hex.append(Character.forDigit((b >>> 4) & 0x0f, 16));
                hex.append(Character.forDigit(b & 0x0f, 16));
            }
            return hex.toString();
        } catch (Exception ignored) {
            return Integer.toHexString(value.hashCode());
        }
    }

    private static boolean gemma4LiteRtMatches(String requestedKey, Path candidate) {
        String candidateKey = gemma4EquivalentKey(candidate.getFileName().toString());
        return !candidateKey.isBlank()
                && (requestedKey.contains(candidateKey) || candidateKey.contains(requestedKey));
    }

    private static int gemma4LiteRtPreference(Path candidate) {
        String name = candidate.getFileName().toString().toLowerCase(Locale.ROOT);
        int score = 0;
        if (name.contains("qualcomm")) {
            score += 100;
        }
        if (!name.endsWith(".litertlm")) {
            score += 20;
        }
        return score;
    }

    private static boolean isLiteRtLmFile(Path path) {
        String name = path.getFileName().toString().toLowerCase(Locale.ROOT);
        return name.endsWith(".litertlm");
    }

    private static String gemma4EquivalentKey(String value) {
        if (value == null || value.isBlank()) {
            return "";
        }
        String normalized = value.toLowerCase(Locale.ROOT)
                .replace("__", "/")
                .replace('\\', '/')
                .replaceAll("\\.(litertlm|litert|tflite|tfl|task|gguf|safetensors?|bin)$", " ")
                .replace("qat", " ")
                .replace("mobile", " ")
                .replace("transformers", " ")
                .replace("safetensor", " ")
                .replace("safetensors", " ")
                .replace("google", " ")
                .replace("litert", " ")
                .replace("community", " ")
                .replace("qualcomm", " ");
        StringBuilder key = new StringBuilder();
        for (String token : normalized.split("[^a-z0-9]+")) {
            if (!token.isBlank()) {
                key.append(token);
            }
        }
        return key.toString();
    }

    private static boolean isGemma4MobileQat(Path modelDir) {
        ModelFamilyQuantizedLoaderProfile profile = ModelFamilyQuantizedLoaderProfile.fromModelDir(modelDir);
        return profile != null && profile.mobileQatSupported();
    }

    private static boolean looksLikeLiteRtArtifactOrDirectory(String localPath) {
        try {
            Path path = Path.of(localPath).toAbsolutePath().normalize();
            if (Files.isRegularFile(path)) {
                return isLiteRtFileName(path.getFileName().toString());
            }
            if (!Files.isDirectory(path)) {
                return false;
            }
            try (var stream = Files.list(path)) {
                return stream
                        .filter(Files::isRegularFile)
                        .map(candidate -> candidate.getFileName().toString())
                        .anyMatch(DirectSafetensorRoutePolicy::isLiteRtFileName);
            }
        } catch (Exception ignored) {
            return false;
        }
    }

    private static boolean isLiteRtFileName(String name) {
        return name != null
                && (name.toLowerCase(Locale.ROOT).endsWith(".litertlm")
                || name.toLowerCase(Locale.ROOT).endsWith(".task")
                || name.toLowerCase(Locale.ROOT).endsWith(".tflite"));
    }

    private static Optional<Path> modelDirectory(String localPath) {
        try {
            Path path = Path.of(localPath);
            Path modelDir = Files.isDirectory(path) ? path : path.getParent();
            return modelDir != null && Files.isDirectory(modelDir)
                    ? Optional.of(modelDir)
                    : Optional.empty();
        } catch (Exception ignored) {
            return Optional.empty();
        }
    }

    private static boolean isGemma4Unified(DirectSafetensorRunProfile profile) {
        return profile != null && profile.unifiedMultimodal();
    }

    private static Map<String, Object> mergeDetails(
            Map<String, Object> base,
            Map<String, Object> additions) {
        Map<String, Object> merged = new LinkedHashMap<>();
        if (base != null) {
            merged.putAll(base);
        }
        if (additions != null) {
            merged.putAll(additions);
        }
        return Map.copyOf(merged);
    }

    private static Optional<Path> findPreferredAlternateArtifact(Path dir, String requestedModelId, String... suffixes) {
        if (dir == null || suffixes == null || suffixes.length == 0) {
            return Optional.empty();
        }
        String modelHint = normalizeArtifactHint(requestedModelId);
        try (var stream = Files.walk(dir, 2)) {
            Optional<Path> preferred = stream.filter(Files::isRegularFile)
                    .filter(p -> hasAnySuffix(p, suffixes))
                    .filter(p -> isCompatibleAlternateArtifact(p, modelHint))
                    .findFirst();
            if (preferred.isPresent()) {
                return preferred;
            }
            if (modelHint == null || modelHint.isBlank()) {
                return findFirstRegularFile(dir, suffixes);
            }
            return Optional.empty();
        } catch (Exception ignored) {
            return Optional.empty();
        }
    }

    private static boolean hasAnySuffix(Path path, String... suffixes) {
        if (path == null || suffixes == null || suffixes.length == 0) {
            return false;
        }
        String name = path.getFileName().toString().toLowerCase(Locale.ROOT);
        for (String suffix : suffixes) {
            if (suffix != null && name.endsWith(suffix.toLowerCase(Locale.ROOT))) {
                return true;
            }
        }
        return false;
    }

    private static boolean isCompatibleAlternateArtifact(Path candidate, String modelHint) {
        String filename = candidate.getFileName().toString().toLowerCase(Locale.ROOT);
        if (filename.contains("tiny_garden")) {
            return false;
        }
        if (modelHint == null || modelHint.isBlank()) {
            return true;
        }
        String normalizedName = normalizeArtifactHint(filename);
        if (normalizedName.contains(modelHint)) {
            return true;
        }
        if (modelHint.contains("functiongemma")) {
            return normalizedName.contains("functiongemma");
        }
        return true;
    }

    private static String normalizeArtifactHint(String value) {
        if (value == null) {
            return "";
        }
        return value.toLowerCase(Locale.ROOT).replaceAll("[^a-z0-9]+", "");
    }

    private static Optional<Path> findFirstRegularFile(Path dir, String... suffixes) {
        if (dir == null || suffixes == null || suffixes.length == 0) {
            return Optional.empty();
        }
        try (var stream = Files.walk(dir, 2)) {
            return stream.filter(Files::isRegularFile)
                    .filter(p -> hasAnySuffix(p, suffixes))
                    .findFirst();
        } catch (Exception ignored) {
            return Optional.empty();
        }
    }

    private static boolean containsAny(String value, String... needles) {
        if (value == null || value.isBlank() || needles == null || needles.length == 0) {
            return false;
        }
        for (String needle : needles) {
            if (needle != null && !needle.isBlank() && value.contains(needle)) {
                return true;
            }
        }
        return false;
    }

    private static boolean hasText(String value) {
        return value != null && !value.isBlank();
    }
}
