package tech.kayys.gollek.inference.libtorch.config;

import io.smallrye.config.WithDefault;

import java.util.List;
import java.util.Optional;

public interface AdvancedConfig {
    /**
     * Master switch for advanced CUDA optimization path.
     */
    @WithDefault("false")
    boolean enabled();

    /**
     * Attention implementation mode.
     * Supported values: baseline, hybrid_fp8_bf16
     */
    @WithDefault("baseline")
    String attentionMode();

    /**
     * Enable FP8 row-wise quantized weight path.
     */
    @WithDefault("false")
    boolean fp8RowwiseEnabled();

    /**
     * Optional tenant allow-list for FP8 rowwise canary mode.
     * Empty = all tenants are eligible.
     */
    Optional<List<String>> fp8RowwiseAllowedTenants();

    /**
     * Optional model allow-list for FP8 rowwise canary mode.
     * Empty = all models are eligible.
     */
    Optional<List<String>> fp8RowwiseAllowedModels();

    /**
     * Optional tenant deny-list for FP8 rowwise canary mode.
     * Deny-list takes precedence over allow-list.
     */
    Optional<List<String>> fp8RowwiseBlockedTenants();

    /**
     * Optional model deny-list for FP8 rowwise canary mode.
     * Deny-list takes precedence over allow-list.
     */
    Optional<List<String>> fp8RowwiseBlockedModels();

    /**
     * Enable SageAttention2-like experimental path.
     */
    @WithDefault("false")
    boolean sageAttention2Enabled();

    /**
     * Optional tenant allow-list for SageAttention2 canary mode.
     * Empty = all tenants are eligible.
     */
    Optional<List<String>> sageAttention2AllowedTenants();

    /**
     * Optional model allow-list for SageAttention2 canary mode.
     * Empty = all models are eligible.
     */
    Optional<List<String>> sageAttention2AllowedModels();

    /**
     * Optional tenant deny-list for SageAttention2 canary mode.
     * Deny-list takes precedence over allow-list.
     */
    Optional<List<String>> sageAttention2BlockedTenants();

    /**
     * Optional model deny-list for SageAttention2 canary mode.
     * Deny-list takes precedence over allow-list.
     */
    Optional<List<String>> sageAttention2BlockedModels();

    /**
     * Allow-list of GPU SM versions for advanced path (comma-separated).
     * Example: "89,90"
     */
    @WithDefault("89,90")
    String allowedGpuSm();
}
