package tech.kayys.gollek.langchain4j;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.output.Response;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import tech.kayys.gollek.sdk.GollekClient;
import tech.kayys.gollek.sdk.model.GenerationRequest;
import tech.kayys.gollek.sdk.model.GenerationResponse;

import java.util.Collections;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.when;

public class GollekChatModelTest {

    @Test
    void should_generate_response() {
        // Given
        GollekClient mockClient = Mockito.mock(GollekClient.class);
        GenerationResponse mockResponse = GenerationResponse.builder()
                .text("Hello from Gollek!")
                .build();
        
        when(mockClient.generate(any(GenerationRequest.class))).thenReturn(mockResponse);

        ChatLanguageModel model = GollekChatModel.builder()
                .client(mockClient)
                .build();

        // When
        String response = model.generate("Hi");

        // Then
        assertThat(response).isEqualTo("Hello from Gollek!");
    }
}
