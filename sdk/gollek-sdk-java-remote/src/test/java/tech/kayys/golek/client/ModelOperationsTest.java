package tech.kayys.gollek.client;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import tech.kayys.gollek.sdk.core.model.ModelInfo;
import tech.kayys.gollek.sdk.core.model.PullProgress;
import tech.kayys.gollek.sdk.core.exception.SdkException;
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

    private GollekClient client;

    @BeforeEach
    void setUp() {
        client = new GollekClient.Builder()
                .baseUrl("http://localhost:8080")
                .apiKey("test-api-key")
                .build();

        // Use reflection to inject the mocked HttpClient
        try {
            java.lang.reflect.Field httpClientField = GollekClient.class.getDeclaredField("httpClient");
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
                "{\"modelId\":\"model1\", \"name\":\"Test Model 1\", \"sizeBytes\":1073741824, \"metadata\":{\"description\":\"First test model\"}},"
                +
                "{\"modelId\":\"model2\", \"name\":\"Test Model 2\", \"sizeBytes\":2147483648, \"metadata\":{\"description\":\"Second test model\"}}"
                +
                "]";

        HttpResponse<String> httpResponse = mock(HttpResponse.class);
        when(httpResponse.statusCode()).thenReturn(200);
        when(httpResponse.body()).thenReturn(jsonResponse);

        when(httpClient.send(any(HttpRequest.class), any(HttpResponse.BodyHandler.class)))
                .thenReturn((HttpResponse) httpResponse);

        // Act
        List<ModelInfo> models = client.listModels();

        // Assert
        assertNotNull(models);
        assertEquals(2, models.size());
        assertEquals("model1", models.get(0).getModelId());
        assertEquals("Test Model 1", models.get(0).getName());
        assertEquals(1073741824L, models.get(0).getSizeBytes());
        assertEquals("model2", models.get(1).getModelId());
        assertEquals("Test Model 2", models.get(1).getName());
        assertEquals(2147483648L, models.get(1).getSizeBytes());
    }

    @Test
    void testListModelsWithPagination_Success() throws Exception {
        // Arrange
        String jsonResponse = "[" +
                "{\"modelId\":\"model3\", \"name\":\"Test Model 3\", \"sizeBytes\":3221225472, \"metadata\":{\"description\":\"Third test model\"}}"
                +
                "]";

        HttpResponse<String> httpResponse = mock(HttpResponse.class);
        when(httpResponse.statusCode()).thenReturn(200);
        when(httpResponse.body()).thenReturn(jsonResponse);

        when(httpClient.send(any(HttpRequest.class), any(HttpResponse.BodyHandler.class)))
                .thenReturn((HttpResponse) httpResponse);

        // Act
        List<ModelInfo> models = client.listModels(10, 5);

        // Assert
        assertNotNull(models);
        assertEquals(1, models.size());
        assertEquals("model3", models.get(0).getModelId());
        assertEquals("Test Model 3", models.get(0).getName());
        assertEquals(3221225472L, models.get(0).getSizeBytes());
    }

    @Test
    void testGetModelInfo_Success() throws Exception {
        // Arrange
        String jsonResponse = "{" +
                "\"modelId\":\"llama3:latest\", \"name\":\"Llama 3 Latest\", \"sizeBytes\":5046586572, " +
                "\"metadata\":{\"description\":\"Latest Llama 3 model\"}" +
                "}";

        HttpResponse<String> httpResponse = mock(HttpResponse.class);
        when(httpResponse.statusCode()).thenReturn(200);
        when(httpResponse.body()).thenReturn(jsonResponse);

        when(httpClient.send(any(HttpRequest.class), any(HttpResponse.BodyHandler.class)))
                .thenReturn((HttpResponse) httpResponse);

        // Act
        Optional<ModelInfo> modelInfoOpt = client.getModelInfo("llama3:latest");

        // Assert
        assertTrue(modelInfoOpt.isPresent());
        ModelInfo modelInfo = modelInfoOpt.get();
        assertEquals("llama3:latest", modelInfo.getModelId());
        assertEquals("Llama 3 Latest", modelInfo.getName());
        assertEquals(5046586572L, modelInfo.getSizeBytes());
        assertEquals("Latest Llama 3 model", modelInfo.getMetadata().get("description"));
    }

    @Test
    void testGetModelInfo_NotFound() throws Exception {
        // Arrange
        HttpResponse<String> httpResponse = mock(HttpResponse.class);
        when(httpResponse.statusCode()).thenReturn(404);
        when(httpClient.send(any(HttpRequest.class), any(HttpResponse.BodyHandler.class)))
                .thenReturn((HttpResponse) httpResponse);

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
        when(httpClient.send(any(HttpRequest.class), any(HttpResponse.BodyHandler.class)))
                .thenReturn((HttpResponse) httpResponse);

        // Act & Assert (should not throw any exception)
        assertDoesNotThrow(() -> client.deleteModel("test-model-to-delete"));
    }

    @Test
    void testDeleteModel_With204Success() throws Exception {
        // Arrange
        HttpResponse<String> httpResponse = mock(HttpResponse.class);
        when(httpResponse.statusCode()).thenReturn(204); // No content response
        when(httpClient.send(any(HttpRequest.class), any(HttpResponse.BodyHandler.class)))
                .thenReturn((HttpResponse) httpResponse);

        // Act & Assert (should not throw any exception)
        assertDoesNotThrow(() -> client.deleteModel("test-model-to-delete"));
    }

    @Test
    void testDeleteModel_NotFound() throws Exception {
        // Arrange
        HttpResponse<String> httpResponse = mock(HttpResponse.class);
        when(httpResponse.statusCode()).thenReturn(404);
        when(httpResponse.body()).thenReturn("{\"error\":\"Model not found\"}");

        when(httpClient.send(any(HttpRequest.class), any(HttpResponse.BodyHandler.class)))
                .thenReturn((HttpResponse) httpResponse);

        // Act & Assert
        SdkException exception = assertThrows(
                SdkException.class,
                () -> client.deleteModel("nonexistent-model"));
    }

    @Test
    void testPullModel_Success() throws Exception {
        // Arrange
        AtomicReference<PullProgress> capturedProgress = new AtomicReference<>();

        HttpResponse<String> httpResponse = mock(HttpResponse.class);
        when(httpResponse.statusCode()).thenReturn(200); // Immediate success
        when(httpClient.send(any(HttpRequest.class), any(HttpResponse.BodyHandler.class)))
                .thenReturn((HttpResponse) httpResponse);

        // Act
        assertDoesNotThrow(() -> client.pullModel("test-model", progress -> {
            capturedProgress.set(progress);
        }));

        // Assert
        // Since the response is immediate success, the progress callback should be
        // called with completion
        // This is a simplified test - in a real scenario, we'd test the streaming
        // behavior too
    }
}