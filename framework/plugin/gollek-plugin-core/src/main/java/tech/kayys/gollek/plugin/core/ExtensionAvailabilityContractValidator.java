/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 */

package tech.kayys.gollek.plugin.core;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;
import java.util.regex.Pattern;

/**
 * Validates the shared availability contract used by detachable extensions.
 */
public final class ExtensionAvailabilityContractValidator {
    private static final Pattern STABLE_ID = Pattern.compile("[a-z][a-z0-9._-]*");

    private ExtensionAvailabilityContractValidator() {
    }

    public static List<ExtensionAvailabilityContractViolation> validate(
            ExtensionAvailabilityProvider provider,
            ExtensionAvailability availability) {
        if (provider == null) {
            return List.of(new ExtensionAvailabilityContractViolation(
                    "unknown",
                    "provider_null",
                    "extension availability provider must not be null"));
        }

        List<ExtensionAvailabilityContractViolation> violations = new ArrayList<>();
        String providerId = safeProviderText(provider::extensionId, null);
        String providerKind = safeProviderText(provider::extensionKind, null);
        String violationId = safeText(providerId, "unknown");

        validateStableToken(violations, providerId, "provider_id", "extension id");
        validateStableToken(violations, providerKind, "provider_kind", "extension kind");
        if (!hasText(safeProviderText(provider::extensionName, null))) {
            violations.add(new ExtensionAvailabilityContractViolation(
                    violationId,
                    "provider_name_blank",
                    "extension name must not be blank"));
        }

        if (availability == null) {
            violations.add(new ExtensionAvailabilityContractViolation(
                    violationId,
                    "availability_null",
                    "availability provider returned null"));
            return List.copyOf(violations);
        }

        if (hasText(providerId) && !providerId.equals(availability.id())) {
            violations.add(new ExtensionAvailabilityContractViolation(
                    violationId,
                    "availability_id_mismatch",
                    "availability id must match provider id: " + availability.id()));
        }
        if (hasText(providerKind) && !providerKind.equals(availability.kind())) {
            violations.add(new ExtensionAvailabilityContractViolation(
                    violationId,
                    "availability_kind_mismatch",
                    "availability kind must match provider kind: " + availability.kind()));
        }
        if (availability.attached() && availability.detached()) {
            violations.add(new ExtensionAvailabilityContractViolation(
                    violationId,
                    "attached_detached_conflict",
                    "extension cannot be both attached and detached"));
        }
        if (availability.productionReady() && availability.detached()) {
            violations.add(new ExtensionAvailabilityContractViolation(
                    violationId,
                    "production_ready_detached",
                    "detached extension cannot be production-ready"));
        }
        if (availability.productionReady() && !availability.healthy()) {
            violations.add(new ExtensionAvailabilityContractViolation(
                    violationId,
                    "production_ready_unhealthy",
                    "unhealthy extension cannot be production-ready"));
        }
        if ("error".equalsIgnoreCase(availability.status())) {
            violations.add(new ExtensionAvailabilityContractViolation(
                    violationId,
                    "availability_error",
                    "availability provider reported an error state"));
        }

        return List.copyOf(violations);
    }

    private static void validateStableToken(
            List<ExtensionAvailabilityContractViolation> violations,
            String value,
            String codePrefix,
            String label) {
        if (!hasText(value)) {
            violations.add(new ExtensionAvailabilityContractViolation(
                    "unknown",
                    codePrefix + "_blank",
                    label + " must not be blank"));
            return;
        }
        if (!STABLE_ID.matcher(value).matches()) {
            violations.add(new ExtensionAvailabilityContractViolation(
                    value,
                    codePrefix + "_invalid",
                    label + " must match " + STABLE_ID.pattern()));
        }
    }

    private static String safeText(String value, String fallback) {
        return hasText(value) ? value.trim() : fallback;
    }

    private static String safeProviderText(Supplier<String> supplier, String fallback) {
        try {
            String value = supplier.get();
            return value == null ? fallback : value.trim();
        } catch (LinkageError | RuntimeException e) {
            return fallback;
        }
    }

    private static boolean hasText(String value) {
        return value != null && !value.isBlank();
    }
}
