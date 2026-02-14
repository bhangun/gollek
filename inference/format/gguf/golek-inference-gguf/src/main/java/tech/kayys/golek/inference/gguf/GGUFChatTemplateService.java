package tech.kayys.golek.inference.gguf;

import com.hubspot.jinjava.Jinjava;
import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.golek.spi.Message;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.Collectors;

@ApplicationScoped
public class GGUFChatTemplateService {

    private static final Logger LOG = Logger.getLogger(GGUFChatTemplateService.class.getName());

    private Jinjava jinjava;
    private boolean jinjavaAvailable = false;
    private boolean initialized = false;

    public GGUFChatTemplateService() {
        // Lazy init: don't create Jinjava here, wait until first render()
        // so the CLI has time to configure logger levels via --logs flag
    }

    private synchronized void initJinjava() {
        if (initialized)
            return;
        initialized = true;
        try {
            this.jinjava = new Jinjava();
            this.jinjavaAvailable = true;
        } catch (Throwable e) {
            LOG.log(Level.FINE,
                    "Jinjava template engine failed to initialize (likely due to native image restrictions). Using fallback ChatML renderer. Error: "
                            + e.getMessage());
            this.jinjava = null;
            this.jinjavaAvailable = false;
        }
    }

    public String render(String template, List<Message> messages) {
        if (!initialized) {
            initJinjava();
        }
        if (!jinjavaAvailable || template == null || template.isBlank()) {
            return fallbackRender(messages);
        }

        try {
            Map<String, Object> context = new HashMap<>();
            context.put("messages", convertMessages(messages));

            // Add common template helper flags
            context.put("add_generation_prompt", true);
            context.put("bos_token", ""); // LlamaCpp handles BOS
            context.put("eos_token", ""); // LlamaCpp handles EOS

            return jinjava.render(template, context);
        } catch (Exception e) {
            // Fallback if rendering fails
            LOG.log(Level.FINE, "Template rendering failed, using fallback: " + e.getMessage());
            return fallbackRender(messages);
        }
    }

    private List<Map<String, String>> convertMessages(List<Message> messages) {
        return messages.stream()
                .map(m -> {
                    Map<String, String> map = new HashMap<>();
                    map.put("role", m.getRole().name().toLowerCase());
                    map.put("content", m.getContent());
                    return map;
                })
                .collect(Collectors.toList());
    }

    private String fallbackRender(List<Message> messages) {
        // Simple ChatML-like fallback
        StringBuilder sb = new StringBuilder();
        for (Message msg : messages) {
            sb.append("<|im_start|>").append(msg.getRole().name().toLowerCase()).append("\n");
            sb.append(msg.getContent()).append("<|im_end|>\n");
        }
        sb.append("<|im_start|>assistant\n");
        return sb.toString();
    }
}
