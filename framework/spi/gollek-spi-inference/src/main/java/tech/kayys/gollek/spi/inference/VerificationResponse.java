package tech.kayys.gollek.spi.inference;

import java.util.List;

/**
 * Response from verifying speculative tokens.
 */
public class VerificationResponse {
    private final List<Integer> acceptedTokens;
    private final int numAccepted;

    public VerificationResponse(List<Integer> acceptedTokens) {
        this.acceptedTokens = acceptedTokens;
        this.numAccepted = acceptedTokens.size();
    }

    public List<Integer> getAcceptedTokens() {
        return acceptedTokens;
    }

    public int getNumAccepted() {
        return numAccepted;
    }
}
