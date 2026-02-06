package tech.kayys.golek.provider.huggingface;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.mockito.ArgumentCaptor;
import tech.kayys.golek.model.download.DownloadProgressListener;

import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

class HuggingFaceClientTest {

    private HttpClient httpClient;
    private HuggingFaceClient client;

    @TempDir
    Path tempDir;

    @BeforeEach
    void setUp() {
        httpClient = mock(HttpClient.class);
        client = new HuggingFaceClient(httpClient);
    }

    @Test
    @SuppressWarnings("unchecked")
    void testGetModelMetadata() throws Exception {
        String modelId = "org/model";
        String json = "{\"id\":\"org/model\", \"siblings\":[{\"rfilename\":\"model.gguf\"}]}";

        HttpResponse<String> response = mock(HttpResponse.class);
        when(response.statusCode()).thenReturn(200);
        when(response.body()).thenReturn(json);

        when(httpClient.send(any(HttpRequest.class), any(HttpResponse.BodyHandler.class)))
                .thenReturn(response);

        Map<String, Object> metadata = client.getModelMetadata(modelId, Optional.empty());

        assertNotNull(metadata);
        assertEquals("org/model", metadata.get("id"));

        ArgumentCaptor<HttpRequest> requestCaptor = ArgumentCaptor.forClass(HttpRequest.class);
        verify(httpClient).send(requestCaptor.capture(), any());
        assertTrue(requestCaptor.getValue().uri().toString().contains(modelId));
    }

    @Test
    @SuppressWarnings("unchecked")
    void testDownloadFile() throws Exception {
        String modelId = "org/model";
        String filename = "model.gguf";
        Path targetPath = tempDir.resolve(filename);
        byte[] content = "fake model data".getBytes();

        HttpResponse<InputStream> response = mock(HttpResponse.class);
        when(response.statusCode()).thenReturn(200);
        when(response.body()).thenReturn(new ByteArrayInputStream(content));

        java.net.http.HttpHeaders headers = java.net.http.HttpHeaders.of(
                Map.of("Content-Length", List.of(String.valueOf(content.length))),
                (k, v) -> true);
        when(response.headers()).thenReturn(headers);

        when(httpClient.send(any(HttpRequest.class), any(HttpResponse.BodyHandler.class)))
                .thenReturn(response);

        DownloadProgressListener listener = mock(DownloadProgressListener.class);

        client.downloadFile(modelId, filename, targetPath, Optional.empty(), listener);

        assertTrue(Files.exists(targetPath));
        assertArrayEquals(content, Files.readAllBytes(targetPath));

        verify(listener).onStart(content.length);
        verify(listener, atLeastOnce()).onProgress(anyLong(), anyLong(), anyDouble());
        verify(listener).onComplete(content.length);
    }

    @Test
    @SuppressWarnings("unchecked")
    void testDownloadRange() throws Exception {
        String modelId = "org/model";
        String filename = "model.gguf";
        byte[] chunkContent = "chunk data".getBytes();

        HttpResponse<InputStream> response = mock(HttpResponse.class);
        when(response.statusCode()).thenReturn(206);
        when(response.body()).thenReturn(new ByteArrayInputStream(chunkContent));

        when(httpClient.send(any(HttpRequest.class), any(HttpResponse.BodyHandler.class)))
                .thenReturn(response);

        InputStream result = client.downloadRange(modelId, filename, 0, 9, Optional.of("token"));

        assertNotNull(result);
        assertArrayEquals(chunkContent, result.readAllBytes());

        ArgumentCaptor<HttpRequest> requestCaptor = ArgumentCaptor.forClass(HttpRequest.class);
        verify(httpClient).send(requestCaptor.capture(), any());
        HttpRequest request = requestCaptor.getValue();
        assertEquals("bytes=0-9", request.headers().firstValue("Range").orElse(null));
        assertEquals("Bearer token", request.headers().firstValue("Authorization").orElse(null));
    }
}
