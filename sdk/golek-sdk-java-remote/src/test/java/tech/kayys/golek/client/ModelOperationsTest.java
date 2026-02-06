package tech.kayys.golek.client;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import tech.kayys.golek.sdk.core.model.ModelInfo;
import tech.kayys.golek.sdk.core.model.PullProgress;
import tech.kayys.golek.client.exception.GolekClientException;

import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class ModelOperationsTest {

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
    void testListModels_Success() throws Exception {
        // Arrange
        String jsonResponse = "[" +
                "{\"id\":\"model1\", \"name\":\"Test Model 1\", \"size\":\"1GB\", \"description\":\"First test model\"}," +
                "{\"id\":\"model2\", \"name\":\"Test Model 2\", \"size\":\"2GB\", \"description\":\"Second test model\"}" +
                "]";

        HttpResponse<String> httpResponse = mock(HttpResponse.class);
        when(httpResponse.statusCode()).thenReturn(200);
        when(httpResponse.body()).thenReturn(jsonResponse);

        when(httpClient.send(any(HttpRequest.class), any())).thenReturn(httpResponse);

        // Act
        List<ModelInfo> models = client.listModels();

        // Assert
        assertNotNull(models);
        assertEquals(2, models.size());
        assertEquals("model1", models.get(0).getId());
        assertEquals("Test Model 1", models.get(0).getName());
        assertEquals("1GB", models.get(0).getSize());
        assertEquals("model2", models.get(1).getId());
        assertEquals("Test Model 2", models.get(1).getName());
        assertEquals("2GB", models.get(1).getSize());
    }

    @Test
    void testListModelsWithPagination_Success() throws Exception {
        // Arrange
        String jsonResponse = "[" +
                "{\"id\":\"model3\", \"name\":\"Test Model 3\", \"size\":\"3GB\", \"description\":\"Third test model\"}" +
                "]";

        HttpResponse<String> httpResponse = mock(HttpResponse.class);
        when(httpResponse.statusCode()).thenReturn(200);
        when(httpResponse.body()).thenReturn(jsonResponse);

        when(httpClient.send(any(HttpRequest.class), any())).thenReturn(httpResponse);

        // Act
        List<ModelInfo> models = client.listModels(10, 5);

        // Assert
        assertNotNull(models);
        assertEquals(1, models.size());
        assertEquals("model3", models.get(0).getId());
        assertEquals("Test Model 3", models.get(0).getName());
        assertEquals("3GB", models.get(0).getSize());
    }

    @Test
    void testGetModelInfo_Success() throws Exception {
        // Arrange
        String jsonResponse = "{" +
                "\"id\":\"llama3:latest\", \"name\":\"Llama 3 Latest\", \"size\":\"4.7GB\", " +
                "\"description\":\"Latest Llama 3 model\"" +
                "}";

        HttpResponse<String> httpResponse = mock(HttpResponse.class);
        when(httpResponse.statusCode()).thenReturn(200);
        when(httpResponse.body()).thenReturn(jsonResponse);

        when(httpClient.send(any(HttpRequest.class), any())).thenReturn(httpResponse);

        // Act
        Optional<ModelInfo> modelInfoOpt = client.getModelInfo("llama3:latest");

        // Assert
        assertTrue(modelInfoOpt.isPresent());
        ModelInfo modelInfo = modelInfoOpt.get();
        assertEquals("llama3:latest", modelInfo.getId());
        assertEquals("Llama 3 Latest", modelInfo.getName());
        assertEquals("4.7GB", modelInfo.getSize());
        assertEquals("Latest Llama 3 model", modelInfo.getDescription());
    }

    @Test
    void testGetModelInfo_NotFound() throws Exception {
        // Arrange
        HttpResponse<String> httpResponse = mock(HttpResponse.class);
        when(httpResponse.statusCode()).thenReturn(404);
        when(httpResponse.body()).thenReturn("{\"error\":\"Model not found\"}");

        when(httpClient.send(any(HttpRequest.class), any())).thenReturn(httpResponse);

        // Act
        Optional<ModelInfo> modelInfoOpt = client.getModelInfo("nonexistent-model");

        // Assert
        assertFalse(modelInfoOpt.isPresent());
    }

    @Test
    void testDeleteModel_Success() throws Exception {
        // Arrange
        HttpResponse<String> httpResponse = mock(HttpResponse.class);
        when(httpResponse.statusCode()).thenReturn(200);
        when(httpResponse.body()).thenReturn("{}"); // Empty response body for successful deletion

        when(httpClient.send(any(HttpRequest.class), any())).thenReturn(httpResponse);

        // Act & Assert (should not throw any exception)
        assertDoesNotThrow(() -> client.deleteModel("test-model-to-delete"));
    }

    @Test
    void testDeleteModel_With204Success() throws Exception {
        // Arrange
        HttpResponse<String> httpResponse = mock(HttpResponse.class);
        when(httpResponse.statusCode()).thenReturn(204); // No content response
        when(httpResponse.body()).thenReturn(""); // Empty response body

        when(httpClient.send(any(HttpRequest.class), any())).thenReturn(httpResponse);

        // Act & Assert (should not throw any exception)
        assertDoesNotThrow(() -> client.deleteModel("test-model-to-delete"));
    }

    @Test
    void testDeleteModel_NotFound() {
        // Arrange
        HttpResponse<String> httpResponse = mock(HttpResponse.class);
        when(httpResponse.statusCode()).thenReturn(404);
        when(httpResponse.body()).thenReturn("{\"error\":\"Model not found\"}");

        when(httpClient.send(any(HttpRequest.class), any())).thenThrow(new RuntimeException("Simulated network error"));

        // Act & Assert
        assertThrows(Exception.class, () -> client.deleteModel("nonexistent-model"));
    }

    @Test
    void testPullModel_Success() throws Exception {
        // Arrange
        AtomicReference<PullProgress> capturedProgress = new AtomicReference<>();

        HttpResponse<String> httpResponse = mock(HttpResponse.class);
        when(httpResponse.statusCode()).thenReturn(200); // Immediate success
        when(httpResponse.body()).thenReturn("{\"status\":\"success\", \"message\":\"Model pulled\"}");

        when(httpClient.send(any(HttpRequest.class), any())).thenReturn(httpResponse);

        // Act
        assertDoesNotThrow(() -> client.pullModel("test-model", progress -> {
            capturedProgress.set(progress);
        }));

        // Assert
        // Since the response is immediate success, the progress callback should be called with completion
        // This is a simplified test - in a real scenario, we'd test the streaming behavior too
    }
}