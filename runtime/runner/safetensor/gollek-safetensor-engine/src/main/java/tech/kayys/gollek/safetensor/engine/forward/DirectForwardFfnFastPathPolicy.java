/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

final class DirectForwardFfnFastPathPolicy {
    private static final DirectForwardFfnFastPathOptions OPTIONS =
            DirectForwardFfnFastPathOptions.fromSystemProperties();
    private static final DirectForwardFfnFastPathRoutingPolicy ROUTING =
            DirectForwardFfnFastPathRoutingPolicy.from(OPTIONS);

    private DirectForwardFfnFastPathPolicy() {
    }

    static boolean isMetalFusedFfnDisabled() {
        return OPTIONS.disableMetalFusedFfn();
    }

    static boolean shouldUseMetalGegluFusedFfn(ModelConfigTraits traits) {
        return ROUTING.shouldUseMetalGegluFusedFfn(traits);
    }

    static boolean shouldTryLocalFusedHalfFfn(ModelConfigTraits traits) {
        return ROUTING.shouldTryLocalFusedHalfFfn(traits);
    }

    static boolean shouldUseSiluGatedFusedFfn() {
        return OPTIONS.enableSiluGatedFusedFfn();
    }

    static boolean isNativeBf16FusedHalfFfnAllowed() {
        return ROUTING.isNativeBf16FusedHalfFfnAllowed();
    }

    static boolean allowNativeBf16FusedHalfFfn(long rows, ModelConfigTraits traits) {
        return ROUTING.allowNativeBf16FusedHalfFfn(rows, traits);
    }

    static boolean shouldUseMetalFusedFfnPrefill(ModelConfigTraits traits) {
        return ROUTING.shouldUseMetalFusedFfnPrefill(traits);
    }

    static boolean shouldUseMetalGegluMatvecFfn(ModelConfigTraits traits) {
        return ROUTING.shouldUseMetalGegluMatvecFfn(traits);
    }

    static boolean shouldUseMetalSwigluMatvecFfn(ModelConfigTraits traits) {
        return ROUTING.shouldUseMetalSwigluMatvecFfn(traits);
    }

    static boolean shouldUseMetalGateUpMatvecFfn() {
        return ROUTING.shouldUseMetalGateUpMatvecFfn();
    }

    static boolean shouldUseMetalMatvecFfnPrefillRows(ModelConfigTraits traits, long rows) {
        return ROUTING.shouldUseMetalMatvecFfnPrefillRows(traits, rows);
    }

    static boolean shouldPreferMetalFusedFfnPrefillOverMatvecRows(ModelConfigTraits traits, long rows) {
        return ROUTING.shouldPreferMetalFusedFfnPrefillOverMatvecRows(traits, rows);
    }

    static boolean shouldValidateMetalMatvecFfn(boolean traceFfnFastPath) {
        return ROUTING.shouldValidateMetalMatvecFfn(traceFfnFastPath);
    }
}
