package tech.kayys.gollek.runtime.weight;


import tech.kayys.gollek.error.ErrorCode;
import tech.kayys.gollek.spi.exception.InferenceException;import tech.kayys.gollek.core.weight.WeightStore;
import tech.kayys.alkhawarizm.core.tensor.Tensor;
import tech.kayys.alkhawarizm.core.backend.ComputeBackend;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.*;

/**
 * A remote weight store that downloads weights on-demand to a local cache.
 * Once downloaded, it leverages LazyWeightStore to load them agnostically.
 */
public class RemoteWeightStore implements WeightStore, AutoCloseable {
    private final String baseUrl;
    private final Path cacheDir;
    private final LazyWeightStore localStore;
    private final HttpClient httpClient;
    private final ComputeBackend backend;

    public RemoteWeightStore(String baseUrl, Path cacheDir, ComputeBackend backend) throws IOException {
        this.baseUrl = baseUrl.endsWith("/") ? baseUrl : baseUrl + "/";
        this.cacheDir = cacheDir;
        this.backend = backend;
        this.httpClient = HttpClient.newBuilder()
                .followRedirects(HttpClient.Redirect.NORMAL)
                .build();
        
        if (!Files.exists(cacheDir)) {
            Files.createDirectories(cacheDir);
        }
        
        this.localStore = new LazyWeightStore();
        this.localStore.discover(cacheDir, backend);
    }

    @Override
    public Tensor get(String key) {
        if (localStore.contains(key)) {
            return localStore.get(key);
        }
        
        // Try to download. We assume individual .bin files for now, 
        // but this could be extended to fetch a manifest and download sharded Safetensors.
        downloadWeight(key);
        
        // Refresh discovery after download
        try {
            localStore.discover(cacheDir, backend);
        } catch (IOException e) {
            throw new InferenceException(ErrorCode.INTERNAL_ERROR, "Failed to refresh local store", e);
        }
        
        return localStore.get(key);
    }

    private void downloadWeight(String key) {
        // Try common extensions
        String[] extensions = {".bin", ".safetensors"};
        for (String ext : extensions) {
            String fileName = key + ext;
            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(baseUrl + fileName))
                    .GET()
                    .build();
            try {
                HttpResponse<Path> response = httpClient.send(request, 
                        HttpResponse.BodyHandlers.ofFile(cacheDir.resolve(fileName)));
                if (response.statusCode() == 200) {
                    return;
                }
                // Cleanup failed download
                Files.deleteIfExists(cacheDir.resolve(fileName));
            } catch (IOException | InterruptedException ignored) {
            }
        }
        throw new MissingWeightException("Failed to download weight " + key + " from " + baseUrl);
    }

    @Override
    public boolean contains(String key) {
        return localStore.contains(key);
    }

    @Override
    public void close() throws Exception {
        localStore.close();
    }
}
