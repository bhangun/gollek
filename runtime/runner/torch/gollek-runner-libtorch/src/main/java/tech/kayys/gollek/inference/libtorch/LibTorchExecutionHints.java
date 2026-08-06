package tech.kayys.gollek.inference.libtorch;

/**
 * Execution hints passed from provider orchestration to lower-level generation
 * loops.
 */
public record LibTorchExecutionHints(
        boolean hybridFp8Bf16AttentionEnabled,
        boolean sageAttention2Requested,
        boolean sageAttention2Enabled,
        String sageAttention2Reason,
        boolean fp8RowwiseEnabled,
        String fp8RowwiseReason,
        int fp8RowwiseScaleCount,
        double fp8RowwiseScaleMean,
        String fp8RowwiseCalibrationSource,
        float[] fp8RowwiseScales) {

    public static LibTorchExecutionHints baseline() {
        return new LibTorchExecutionHints(false, false, false, "none", false, "baseline", 0, 0.0d, "none", null);
    }

    public static LibTorchExecutionHints baselineWithSageState(
            boolean sageAttention2Requested,
            boolean sageAttention2Enabled,
            String sageAttention2Reason) {
        return new LibTorchExecutionHints(
                false,
                sageAttention2Requested,
                sageAttention2Enabled,
                sageAttention2Reason == null || sageAttention2Reason.isBlank() ? "none" : sageAttention2Reason,
                false,
                "baseline",
                0,
                0.0d,
                "none",
                null);
    }

    public static LibTorchExecutionHints hybridFp8Bf16(
            boolean sageAttention2Requested,
            boolean sageAttention2Enabled,
            String sageAttention2Reason,
            boolean fp8RowwiseEnabled,
            String fp8RowwiseReason,
            int fp8RowwiseScaleCount,
            double fp8RowwiseScaleMean,
            String fp8RowwiseCalibrationSource,
            float[] fp8RowwiseScales) {
        return new LibTorchExecutionHints(
                true,
                sageAttention2Requested,
                sageAttention2Enabled,
                sageAttention2Reason,
                fp8RowwiseEnabled,
                fp8RowwiseReason,
                fp8RowwiseScaleCount,
                fp8RowwiseScaleMean,
                fp8RowwiseCalibrationSource,
                fp8RowwiseScales);
    }
}
