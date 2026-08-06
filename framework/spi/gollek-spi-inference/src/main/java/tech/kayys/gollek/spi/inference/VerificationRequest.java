package tech.kayys.gollek.spi.inference;

import java.util.List;

/**
 * Request to verify speculative tokens proposed by a draft model.
 */
public class VerificationRequest {
    private final String targetModelId;
    private final List<Integer> draftTokens;
    private final List<Integer> prefixTokens;

    public VerificationRequest(String targetModelId, List<Integer> prefixTokens, List<Integer> draftTokens) {
        this.targetModelId = targetModelId;
        this.prefixTokens = prefixTokens;
        this.draftTokens = draftTokens;
    }

    public String getTargetModelId() {
        return targetModelId;
    }

    public List<Integer> getDraftTokens() {
        return draftTokens;
    }

    public List<Integer> getPrefixTokens() {
        return prefixTokens;
    }
}
