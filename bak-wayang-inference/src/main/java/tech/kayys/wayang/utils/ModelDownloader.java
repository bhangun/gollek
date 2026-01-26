package tech.kayys.golek.utils;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.Response;
import okhttp3.ResponseBody;


public class ModelDownloader {
    private static final Logger log = LoggerFactory.getLogger(ModelDownloader.class);
    private final OkHttpClient client;
    private final Path downloadDir;
    
    public ModelDownloader(Path downloadDir) {
        this.downloadDir = downloadDir;
        this.client = new OkHttpClient.Builder()
            .connectTimeout(30, TimeUnit.SECONDS)
            .readTimeout(300, TimeUnit.SECONDS)
            .build();
    }
    
    public Path downloadFromHuggingFace(String repoId, String filename, 
                                        Consumer<DownloadProgress> progressCallback) throws IOException {
        String url = String.format("https://huggingface.co/%s/resolve/main/%s", repoId, filename);
        Path outputPath = downloadDir.resolve(filename);
        
        if (Files.exists(outputPath)) {
            log.info("Model already exists: {}", outputPath);
            return outputPath;
        }
        
        log.info("Downloading model from: {}", url);
        Files.createDirectories(downloadDir);
        
        Request request = new Request.Builder().url(url).build();
        
        try (Response response = client.newCall(request).execute()) {
            if (!response.isSuccessful()) {
                throw new IOException("Download failed: " + response.code());
            }
            
            ResponseBody body = response.body();
            if (body == null) {
                throw new IOException("Empty response body");
            }
            
            long contentLength = body.contentLength();
            
            try (InputStream input = body.byteStream();
                 OutputStream output = Files.newOutputStream(outputPath)) {
                
                byte[] buffer = new byte[8192];
                long downloaded = 0;
                int read;
                
                while ((read = input.read(buffer)) != -1) {
                    output.write(buffer, 0, read);
                    downloaded += read;
                    
                    if (progressCallback != null && contentLength > 0) {
                        double progress = (double) downloaded / contentLength;
                        progressCallback.accept(new DownloadProgress(downloaded, contentLength, progress));
                    }
                }
            }
        }
        
        log.info("Download completed: {}", outputPath);
        return outputPath;
    }
    
    public record DownloadProgress(long downloaded, long total, double percentage) {}
}
