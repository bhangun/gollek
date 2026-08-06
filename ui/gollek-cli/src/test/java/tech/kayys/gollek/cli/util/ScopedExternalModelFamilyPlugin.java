/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 */

package tech.kayys.gollek.cli.util;

import tech.kayys.alkhawarizm.spi.model.ModelFamilyCapability;
import tech.kayys.alkhawarizm.spi.model.ModelFamilyDescriptor;
import tech.kayys.alkhawarizm.spi.model.ModelFamilyPlugin;

import java.util.List;
import java.util.Map;

/**
 * Public test fixture discovered only when an external service-entry jar points to it.
 */
public final class ScopedExternalModelFamilyPlugin implements ModelFamilyPlugin {
    public static final String FAMILY_ID = "scoped_external_family";
    public static final String MODEL_TYPE = "scoped_external";

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                FAMILY_ID,
                "Scoped External Family",
                List.of(MODEL_TYPE),
                List.of("ScopedExternalForCausalLM"),
                List.of(ModelFamilyCapability.CAUSAL_LM),
                Map.of(
                        "bundle_profile", "optional",
                        "origin", "external/model-family/" + FAMILY_ID,
                        "version", "0.1.0-test"));
    }
}
