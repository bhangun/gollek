/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 */

package tech.kayys.gollek.plugin.core;

import java.util.Comparator;
import java.util.Collection;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.ServiceConfigurationError;
import java.util.ServiceLoader;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Registry for extension availability providers discovered through plugin-core.
 */
public final class ExtensionAvailabilityRegistry {
    private static final ExtensionAvailabilityRegistry GLOBAL = new ExtensionAvailabilityRegistry();

    private final Map<String, ExtensionAvailabilityProvider> providers = new ConcurrentHashMap<>();

    public static ExtensionAvailabilityRegistry global() {
        return GLOBAL;
    }

    public static ExtensionAvailabilityRegistry create() {
        return new ExtensionAvailabilityRegistry();
    }

    public ExtensionAvailabilityRegistry snapshot() {
        ExtensionAvailabilityRegistry registry = create();
        registry.registerAll(providers());
        return registry;
    }

    public void register(ExtensionAvailabilityProvider provider) {
        Objects.requireNonNull(provider, "provider");
        providers.put(provider.extensionId(), provider);
    }

    public void registerAll(Collection<? extends ExtensionAvailabilityProvider> providers) {
        if (providers != null) {
            providers.forEach(this::register);
        }
    }

    public void unregister(String extensionId) {
        if (extensionId != null) {
            providers.remove(extensionId);
        }
    }

    public void clear() {
        providers.clear();
    }

    public List<ExtensionAvailabilityProvider> providers() {
        return providers.values().stream()
                .sorted(Comparator.comparing(ExtensionAvailabilityProvider::extensionId))
                .toList();
    }

    public Optional<ExtensionAvailability> availability(String extensionId) {
        ExtensionAvailabilityProvider provider = providers.get(extensionId);
        return provider == null ? Optional.empty() : Optional.of(availabilitySafely(provider));
    }

    public List<ExtensionAvailability> availabilityReports() {
        return providers().stream()
                .map(this::availabilitySafely)
                .toList();
    }

    public List<ExtensionAvailabilityContractViolation> contractViolations() {
        return providers().stream()
                .flatMap(provider -> ExtensionAvailabilityContractValidator
                        .validate(provider, availabilitySafely(provider))
                        .stream())
                .toList();
    }

    public ExtensionAvailabilityContractReport contractReport() {
        return ExtensionAvailabilityContractReport.fromViolations(contractViolations());
    }

    public List<ExtensionAvailabilityProvider> discoverServiceLoaderProviders() {
        ServiceLoader<ExtensionAvailabilityProvider> loader = ServiceLoader.load(ExtensionAvailabilityProvider.class);
        return discoverServiceLoaderProviders(loader);
    }

    public List<ExtensionAvailabilityProvider> discoverServiceLoaderProviders(ClassLoader classLoader) {
        ClassLoader effectiveClassLoader = classLoader == null
                ? Thread.currentThread().getContextClassLoader()
                : classLoader;
        if (effectiveClassLoader == null) {
            effectiveClassLoader = ExtensionAvailabilityProvider.class.getClassLoader();
        }
        ServiceLoader<ExtensionAvailabilityProvider> loader =
                ServiceLoader.load(ExtensionAvailabilityProvider.class, effectiveClassLoader);
        return discoverServiceLoaderProviders(loader);
    }

    private List<ExtensionAvailabilityProvider> discoverServiceLoaderProviders(
            ServiceLoader<ExtensionAvailabilityProvider> loader) {
        Iterator<ExtensionAvailabilityProvider> iterator = loader.iterator();
        while (true) {
            ExtensionAvailabilityProvider provider;
            try {
                if (!iterator.hasNext()) {
                    break;
                }
                provider = iterator.next();
            } catch (ServiceConfigurationError ignored) {
                continue;
            }
            register(provider);
        }
        return providers();
    }

    public ExtensionAvailability availabilitySafely(ExtensionAvailabilityProvider provider) {
        try {
            ExtensionAvailability availability = provider.availability();
            if (availability != null) {
                return availability;
            }
            return errorAvailability(provider, new NullPointerException("availability provider returned null"));
        } catch (LinkageError | RuntimeException e) {
            return errorAvailability(provider, e);
        }
    }

    private ExtensionAvailability errorAvailability(ExtensionAvailabilityProvider provider, Throwable throwable) {
        Map<String, String> attributes = new LinkedHashMap<>();
        attributes.put("errorType", throwable.getClass().getSimpleName());
        String extensionId = safeProviderText(provider, ProviderField.ID, "unknown");
        return new ExtensionAvailability(
                extensionId,
                safeProviderText(provider, ProviderField.NAME, extensionId),
                safeProviderText(provider, ProviderField.KIND, "extension"),
                true,
                false,
                false,
                false,
                "error",
                List.of(),
                List.of(),
                attributes,
                throwable.getMessage(),
                List.of("Inspect extension provider " + extensionId + " before using it in production."));
    }

    private String safeProviderText(
            ExtensionAvailabilityProvider provider,
            ProviderField field,
            String fallback) {
        try {
            String value = switch (field) {
                case ID -> provider.extensionId();
                case NAME -> provider.extensionName();
                case KIND -> provider.extensionKind();
            };
            return value == null || value.isBlank() ? fallback : value;
        } catch (LinkageError | RuntimeException ignored) {
            return fallback;
        }
    }

    private enum ProviderField {
        ID,
        NAME,
        KIND
    }
}
