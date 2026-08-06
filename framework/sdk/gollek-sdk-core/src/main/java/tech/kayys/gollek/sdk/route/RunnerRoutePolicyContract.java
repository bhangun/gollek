package tech.kayys.gollek.sdk.route;

import tech.kayys.gollek.sdk.route.RunnerRoutePolicyFields;

import java.util.Map;

/**
 * Command-package facade for the shared runner-selection policy contract.
 */
public final class RunnerRoutePolicyContract {
    public static final String CONTRACT_ID = RunnerRoutePolicyFields.CONTRACT_ID;
    public static final int SCHEMA_VERSION = RunnerRoutePolicyFields.SCHEMA_VERSION;

    private RunnerRoutePolicyContract() {
    }

    public static Map<String, Object> report() {
        return RunnerRoutePolicyFields.report();
    }

    public static String schemaFingerprint() {
        return RunnerRoutePolicyFields.schemaFingerprint();
    }
}
