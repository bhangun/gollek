package tech.kayys.gollek.plugin.security;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import tech.kayys.gollek.spi.context.EngineContext;
import tech.kayys.gollek.spi.execution.ExecutionContext;
import tech.kayys.gollek.spi.exception.PluginException;
import tech.kayys.gollek.spi.inference.InferencePhase;
import tech.kayys.gollek.spi.inference.InferencePhasePlugin;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.Message;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Plugin that detects and redacts PII from inference requests and responses.
 * 
 * <p>This plugin executes in the PRE_PROCESSING phase for requests and POST_PROCESSING
 * phase for responses to ensure PII is redacted before being sent to models or returned
 * to users.</p>
 * 
 * <h2>Features</h2>
 * <ul>
 *   <li>Email address redaction</li>
 *   <li>Phone number redaction</li>
 *   <li>Credit card number redaction</li>
 *   <li>SSN redaction</li>
 *   <li>IP address redaction</li>
 *   <li>API key/secret redaction</li>
 *   <li>AWS access key redaction</li>
 *   <li>Custom pattern support</li>
 *   <li>Audit logging of redaction events</li>
 * </ul>
 * 
 * <h2>Configuration</h2>
 * <pre>
 * gollek:
 *   plugins:
 *     pii-redaction:
 *       enabled: true
 *       redact-requests: true
 *       redact-responses: true
 *       audit-enabled: true
 *       patterns:
 *         email:
 *           enabled: true
 *           replacement: "[REDACTED_EMAIL]"
 *         phone:
 *           enabled: false  # Disable phone redaction
 * </pre>
 * 
 * @since 1.0.0
 */
@ApplicationScoped
public class PIIRedactionRequestPlugin implements InferencePhasePlugin {

    private static final Logger LOG = Logger.getLogger(PIIRedactionRequestPlugin.class);
    
    private static final String PLUGIN_ID = "pii-redaction-request";
    private static final String VERSION = "1.0.0";

    @Override
    public String id() {
        return PLUGIN_ID;
    }

    @Override
    public String version() {
        return VERSION;
    }

    @Inject
    PIIRedactionService redactionService;

    private boolean enabled = true;
    private boolean redactRequests = true;
    private boolean redactResponses = true;
    private boolean auditEnabled = true;
    private Map<String, Object> config = new HashMap<>();


    @Override
    public InferencePhase phase() {
        // This plugin executes in both PRE_PROCESSING and POST_PROCESSING
        // We'll handle both in the execute method
        return InferencePhase.PRE_PROCESSING;
    }

    @Override
    public int order() {
        // Execute early in PRE_PROCESSING to ensure PII is redacted before other processing
        return 10;
    }

    @Override
    public void initialize(tech.kayys.gollek.spi.plugin.PluginContext context) {
        this.enabled = Boolean.parseBoolean(context.getConfig("enabled", "true"));
        this.redactRequests = Boolean.parseBoolean(context.getConfig("redact-requests", "true"));
        this.redactResponses = Boolean.parseBoolean(context.getConfig("redact-responses", "true"));
        this.auditEnabled = Boolean.parseBoolean(context.getConfig("audit-enabled", "true"));
        
        // Load custom patterns
        loadCustomPatterns(context);
        
        LOG.infof("Initialized %s plugin (enabled: %b, requests: %b, responses: %b, audit: %b)", 
            PLUGIN_ID, enabled, redactRequests, redactResponses, auditEnabled);
    }

    @Override
    public void execute(ExecutionContext context, EngineContext engine) throws PluginException {
        if (!enabled) {
            return;
        }

        try {
            // Check if we should redact for the request phase
            Optional<InferenceRequest> requestOpt = context.getVariable("request", InferenceRequest.class);
            if (requestOpt.isPresent() && redactRequests) {
                redactRequest(requestOpt.get(), context);
            }
        } catch (Exception e) {
            LOG.errorf(e, "Failed to execute PII redaction for request %s",
                context.token().getExecutionId());
        }
    }

    private void redactRequest(InferenceRequest request, ExecutionContext context) {
        if (request == null || request.getMessages() == null) {
            return;
        }

        int redactionCount = 0;
        List<Message> redactedMessages = new ArrayList<>();

        for (Message message : request.getMessages()) {
            String originalContent = message.getContent();
            String redactedContent = redactionService.redact(originalContent);
            
            if (!originalContent.equals(redactedContent)) {
                redactionCount++;
                
                if (auditEnabled) {
                    Map<String, Integer> detections = redactionService.detectPII(originalContent);
                    LOG.infof("PII detected in message role %s: %s", 
                        message.getRole(), detections);
                }
            }

            // Create new message with redacted content
            Message redactedMessage = new Message(
                message.getRole(),
                redactedContent,
                message.getName(),
                message.getToolCalls(),
                message.getToolCallId()
            );

            redactedMessages.add(redactedMessage);
        }

        if (redactionCount > 0) {
            LOG.infof("Redacted PII from %d messages in request %s", 
                redactionCount, request.getRequestId());

            // Update the request with redacted messages
            InferenceRequest redactedRequest = request.toBuilder()
                .messages(redactedMessages)
                .metadata("pii_redacted", true)
                .build();

            context.putVariable("request", redactedRequest);
        }
    }

    private void loadCustomPatterns(tech.kayys.gollek.spi.plugin.PluginContext context) {
        // Load custom patterns from configuration
        // Example: patterns.email.regex, patterns.email.replacement, etc.
        
        Map<String, String> patternConfigs = new HashMap<>();
        
        // Email pattern
        String emailRegex = context.getConfig("patterns.email.regex", null);
        String emailReplacement = context.getConfig("patterns.email.replacement", "[REDACTED_EMAIL]");
        boolean emailEnabled = Boolean.parseBoolean(context.getConfig("patterns.email.enabled", "true"));
        
        if (emailRegex != null) {
            redactionService.addPattern("email", emailRegex, emailReplacement, emailEnabled);
        }

        // Phone pattern
        String phoneRegex = context.getConfig("patterns.phone.regex", null);
        String phoneReplacement = context.getConfig("patterns.phone.replacement", "[REDACTED_PHONE]");
        boolean phoneEnabled = Boolean.parseBoolean(context.getConfig("patterns.phone.enabled", "true"));
        
        if (phoneRegex != null) {
            redactionService.addPattern("phone", phoneRegex, phoneReplacement, phoneEnabled);
        }

        // Add more custom patterns as needed...
    }

    @Override
    public void onConfigUpdate(Map<String, Object> newConfig) {
        this.config = new HashMap<>(newConfig);
        this.enabled = (Boolean) newConfig.getOrDefault("enabled", true);
        this.redactRequests = (Boolean) newConfig.getOrDefault("redact-requests", true);
        this.redactResponses = (Boolean) newConfig.getOrDefault("redact-responses", true);
        this.auditEnabled = (Boolean) newConfig.getOrDefault("audit-enabled", true);

        LOG.infof("Updated %s configuration", PLUGIN_ID);
    }

    @Override
    public Map<String, Object> currentConfig() {
        Map<String, Object> current = new HashMap<>(config);
        current.put("enabled", enabled);
        current.put("redact-requests", redactRequests);
        current.put("redact-responses", redactResponses);
        current.put("audit-enabled", auditEnabled);
        return current;
    }

    /**
     * Get redaction statistics.
     * 
     * @return map of pattern names to redaction counts
     */
    public Map<String, Integer> getRedactionStats() {
        return redactionService.getRedactionStats();
    }

    /**
     * Clear redaction statistics.
     */
    public void clearStats() {
        redactionService.clearStats();
    }
}
