package tech.kayys.gollek.inference.libtorch;

import jakarta.enterprise.context.ApplicationScoped;
import org.jboss.logging.Logger;
import tech.kayys.gollek.spi.Message;

import java.util.List;

/**
 * Chat template rendering service for LibTorch inference.
 * <p>
 * Converts a list of {@link Message} objects into a formatted prompt string
 * using the ChatML template format (compatible with Qwen, Mistral, etc.).
 */
@ApplicationScoped
public class LibTorchChatTemplateService {

    private static final Logger log = Logger.getLogger(LibTorchChatTemplateService.class);

    /**
     * Render messages using ChatML format.
     * <p>
     * Example output:
     * 
     * <pre>
     * &lt;|im_start|&gt;system
     * You are a helpful assistant.&lt;|im_end|&gt;
     * &lt;|im_start|&gt;user
     * Hello!&lt;|im_end|&gt;
     * &lt;|im_start|&gt;assistant
     * </pre>
     *
     * @param messages list of conversation messages
     * @return formatted prompt string
     */
    public String renderChatML(List<Message> messages) {
        if (messages == null || messages.isEmpty()) {
            return "";
        }

        StringBuilder sb = new StringBuilder();

        for (Message msg : messages) {
            String role = resolveRole(msg);
            String content = msg.getContent();

            sb.append("<|im_start|>").append(role).append('\n');
            if (content != null && !content.isBlank()) {
                sb.append(content.strip());
            }
            sb.append("<|im_end|>\n");
        }

        // Add assistant prompt to trigger generation
        sb.append("<|im_start|>assistant\n");

        return sb.toString();
    }

    /**
     * Build a simple prompt without chat template markers.
     * Fallback for models that don't use ChatML.
     */
    public String renderPlain(List<Message> messages) {
        if (messages == null || messages.isEmpty()) {
            return "";
        }

        StringBuilder sb = new StringBuilder();
        for (Message msg : messages) {
            String role = resolveRole(msg);
            String content = msg.getContent();
            if (content != null && !content.isBlank()) {
                sb.append(role.toUpperCase()).append(": ").append(content.strip()).append('\n');
            }
        }
        sb.append("ASSISTANT: ");
        return sb.toString();
    }

    private String resolveRole(Message msg) {
        if (msg.getRole() != null) {
            return switch (msg.getRole()) {
                case SYSTEM -> "system";
                case ASSISTANT -> "assistant";
                case USER -> "user";
                case FUNCTION, TOOL -> "tool";
            };
        }
        return "user";
    }
}
