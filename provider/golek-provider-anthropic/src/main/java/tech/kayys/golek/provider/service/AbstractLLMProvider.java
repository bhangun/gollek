package tech.kayys.golek.provider.service;

import jakarta.inject.Inject;
import org.jboss.logging.Logger;

/**
 * Abstract base for LLM providers.
 */
public abstract class AbstractLLMProvider {

    protected final Logger log;

    @Inject
    protected ProviderMetrics metrics;

    @Inject
    protected ProviderAuditor auditor;

    protected AbstractLLMProvider() {
        this.log = Logger.getLogger(getClass());
    }
}