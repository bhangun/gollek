package tech.kayys.golek.client;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import tech.kayys.golek.api.inference.InferenceRequest;
import tech.kayys.golek.api.inference.InferenceResponse;
import tech.kayys.golek.client.exception.GolekClientException;

import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class GolekClientTest {
    
    @Mock
    private HttpClient httpClient;
    
    private GolekClient client;
    
    @BeforeEach
    void setUp() {
        client = new GolekClient.Builder()
                .baseUrl("http://localhost:8080")
                .apiKey("test-api-key")
                .build();
        
        // Use reflection to inject the mocked HttpClient
        try {
            java.lang.reflect.Field httpClientField = GolekClient.class.getDeclaredField("httpClient");
            httpClientField.setAccessible(true);
            httpClientField.set(client, httpClient);
        } catch (Exception e) {
            throw new RuntimeException("Failed to inject mock HttpClient", e);
        }
    }
    
    @Test
    void testCreateCompletion_Success() throws Exception {
        // Arrange
        InferenceRequest request = InferenceRequest.builder()
                .model("test-model")
                .message(tech.kayys.golek.api.Message.user("Hello"))
                .build();
        
        InferenceResponse expectedResponse = InferenceResponse.builder()
                .requestId(request.getRequestId())
                .content("Test response")
                .model("test-model")
                .build();
        
        String jsonResponse = "{\n" +
                "  \"requestId\": \"" + request.getRequestId() + "\",\n" +
                "  \"content\": \"Test response\",\n" +
                "  \"model\": \"test-model\"\n" +
                "}";
        
        HttpResponse<String> httpResponse = mock(HttpResponse.class);
        when(httpResponse.statusCode()).thenReturn(200);
        when(httpResponse.body()).thenReturn(jsonResponse);
        
        when(httpClient.send(any(HttpRequest.class), any())).thenReturn(httpResponse);
        
        // Act
        InferenceResponse actualResponse = client.createCompletion(request);
        
        // Assert
        assertNotNull(actualResponse);
        assertEquals(expectedResponse.getContent(), actualResponse.getContent());
        assertEquals(expectedResponse.getModel(), actualResponse.getModel());
    }
    
    @Test
    void testCreateCompletion_ApiError() throws Exception {
        // Arrange
        InferenceRequest request = InferenceRequest.builder()
                .model("test-model")
                .message(tech.kayys.golek.api.Message.user("Hello"))
                .build();
        
        HttpResponse<String> httpResponse = mock(HttpResponse.class);
        when(httpResponse.statusCode()).thenReturn(500);
        when(httpResponse.body()).thenReturn("Internal Server Error");
        
        when(httpClient.send(any(HttpRequest.class), any())).thenReturn(httpResponse);
        
        // Act & Assert
        GolekClientException exception = assertThrows(
            GolekClientException.class, 
            () -> client.createCompletion(request)
        );
        
        assertTrue(exception.getMessage().contains("500"));
    }
    
    @Test
    void testCreateCompletion_NetworkError() throws Exception {
        // Arrange
        InferenceRequest request = InferenceRequest.builder()
                .model("test-model")
                .message(tech.kayys.golek.api.Message.user("Hello"))
                .build();
        
        when(httpClient.send(any(HttpRequest.class), any()))
            .thenThrow(new java.io.IOException("Network error"));
        
        // Act & Assert
        GolekClientException exception = assertThrows(
            GolekClientException.class, 
            () -> client.createCompletion(request)
        );
        
        assertTrue(exception.getCause() instanceof java.io.IOException);
    }
    
    @Test
    void testBuilder_WithoutApiKey_ThrowsException() {
        // Act & Assert
        IllegalArgumentException exception = assertThrows(
            IllegalArgumentException.class, 
            () -> new GolekClient.Builder().build()
        );
        
        assertEquals("API key is required", exception.getMessage());
    }
    
    @Test
    void testBuilder_WithEmptyApiKey_ThrowsException() {
        // Act & Assert
        IllegalArgumentException exception = assertThrows(
            IllegalArgumentException.class, 
            () -> new GolekClient.Builder().apiKey("").build()
        );
        
        assertEquals("API key is required", exception.getMessage());
    }
    
    @Test
    void testBuilder_ValidConfiguration_BuildsSuccessfully() {
        // Act
        GolekClient client = new GolekClient.Builder()
                .baseUrl("https://api.example.com")
                .apiKey("valid-api-key")
                .defaultTenantId("test-tenant")
                .connectTimeout(Duration.ofSeconds(10))
                .build();
        
        // Assert
        assertNotNull(client);
    }
}