package tech.kayys.gollek.langchain4j;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.Content;
import dev.langchain4j.data.message.ImageContent;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.TextContent;
import dev.langchain4j.data.message.UserMessage;
import tech.kayys.gollek.sdk.multimodal.ContentPart;

import java.util.ArrayList;
import java.util.List;

/**
 * Mapper between LangChain4j message types and Gollek prompt format.
 */
public class GollekMessageMapper {

    /**
     * Convert a list of LangChain4j messages into a single prompt string.
     * <p>
     * Note: In a production environment, this should use a proper chat template
     * (e.g., Jinja2 for Llama 3 or Mistral). For now, we use a simple bracketed format.
     * </p>
     */
    public static String toPrompt(List<ChatMessage> messages) {
        StringBuilder sb = new StringBuilder();

        for (ChatMessage message : messages) {
            if (message instanceof SystemMessage) {
                sb.append("[SYSTEM] ").append(message.text()).append("\n");
            } else if (message instanceof UserMessage userMessage) {
                sb.append("[USER] ");
                if (userMessage.contents() != null) {
                    for (Content content : userMessage.contents()) {
                        if (content instanceof TextContent) {
                            sb.append(((TextContent) content).text());
                        } else if (content instanceof ImageContent) {
                            sb.append("[IMAGE]");
                        }
                    }
                } else {
                    sb.append(message.text());
                }
                sb.append("\n");
            } else if (message instanceof AiMessage) {
                sb.append("[ASSISTANT] ").append(message.text()).append("\n");
            }
        }

        return sb.toString();
    }

    /**
     * Converts LangChain4j ChatMessages into Gollek ContentParts for multimodal processing.
     */
    public static List<ContentPart> toParts(List<ChatMessage> messages) {
        List<ContentPart> parts = new ArrayList<>();

        for (ChatMessage message : messages) {
            String prefix = "";
            if (message instanceof SystemMessage) prefix = "[SYSTEM] ";
            else if (message instanceof UserMessage) prefix = "[USER] ";
            else if (message instanceof AiMessage) prefix = "[ASSISTANT] ";

            if (message instanceof UserMessage userMessage && userMessage.contents() != null) {
                for (Content content : userMessage.contents()) {
                    if (content instanceof TextContent) {
                        parts.add(ContentPart.text(prefix + ((TextContent) content).text()));
                    } else if (content instanceof ImageContent imageContent) {
                        // ImageContent can have base64 or URL.
                        if (imageContent.image().base64Data() != null) {
                            parts.add(ContentPart.imageUrl("data:image/png;base64," + imageContent.image().base64Data()));
                        } else if (imageContent.image().url() != null) {
                            parts.add(ContentPart.imageUrl(imageContent.image().url().toString()));
                        }
                    }
                }
            } else {
                parts.add(ContentPart.text(prefix + message.text()));
            }
        }

        return parts;
    }
}
