package tech.kayys.gollek.spi.inference;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.spi.Message;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

public class InferenceRequestAttachmentTest {

    @Test
    public void testBuilderWithAttachments() {
        Attachment image = Attachment.fromUrl("https://example.com/image.png", "image/png");
        Attachment audio = Attachment.fromBase64("YmFzZTY0ZGF0YQ==", "audio/wav");

        InferenceRequest request = InferenceRequest.builder()
                .model("gpt-4-vision")
                .message(new Message(Message.Role.USER, "What's in this image?"))
                .attachment(image)
                .attachment(audio)
                .build();

        assertEquals(2, request.getAttachments().size());
        assertEquals(image, request.getAttachments().get(0));
        assertEquals(audio, request.getAttachments().get(1));
        assertEquals("gpt-4-vision", request.getModel());
    }

    @Test
    public void testToBuilderCopiesAttachments() {
        Attachment image = Attachment.fromUrl("https://example.com/image.png", "image/png");
        
        InferenceRequest original = InferenceRequest.builder()
                .model("gpt-4-vision")
                .message(new Message(Message.Role.USER, "Analyze this"))
                .attachment(image)
                .build();

        InferenceRequest copy = original.toBuilder()
                .model("gpt-3.5-turbo")
                .build();

        assertEquals(1, copy.getAttachments().size());
        assertEquals(image, copy.getAttachments().get(0));
        assertEquals("gpt-3.5-turbo", copy.getModel());
        assertEquals(original.getAttachments(), copy.getAttachments());
    }

    @Test
    public void testAttachmentsImmutability() {
        Attachment image = Attachment.fromUrl("https://example.com/image.png", "image/png");
        
        InferenceRequest request = InferenceRequest.builder()
                .model("gpt-4-vision")
                .message(new Message(Message.Role.USER, "Test"))
                .attachment(image)
                .build();

        List<Attachment> attachments = request.getAttachments();
        assertThrows(UnsupportedOperationException.class, () -> 
            attachments.add(Attachment.fromUrl("https://example.com/fail.png", "image/png"))
        );
    }
}
