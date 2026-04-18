package tech.kayys.gollek.multimodal.benchmark;

import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;
import tech.kayys.gollek.spi.model.*;
import tech.kayys.gollek.plugin.runner.llamacpp.processor.LlamaCppMultimodalProcessor;

import java.util.concurrent.TimeUnit;

/**
 * Performance benchmarks for GGUF/LLaVA multimodal processor.
 * 
 * Run with: mvn test -Dtest=Benchmarks
 * Or: java -jar target/benchmarks.jar
 */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@Warmup(iterations = 3, time = 10, timeUnit = TimeUnit.SECONDS)
@Measurement(iterations = 5, time = 10, timeUnit = TimeUnit.SECONDS)
@Fork(2)
@Threads(1)
public class LlamaCppMultimodalBenchmark {

    private LlamaCppMultimodalProcessor processor;
    private byte[] testImage;
    private MultimodalRequest singleRequest;
    private MultimodalRequest detailedRequest;

    @Setup
    public void setUp() throws Exception {
        processor = new LlamaCppMultimodalProcessor();
        testImage = createTestImage();
        
        // Simple request
        singleRequest = MultimodalRequest.builder()
            .model("llava-13b-gguf")
            .inputs(
                MultimodalContent.ofText("What is this?"),
                MultimodalContent.ofBase64Image(testImage, "image/jpeg")
            )
            .build();
        
        // Detailed request with config
        detailedRequest = MultimodalRequest.builder()
            .model("llava-13b-gguf")
            .inputs(
                MultimodalContent.ofText("Describe this image in detail, including objects, colors, and context"),
                MultimodalContent.ofBase64Image(testImage, "image/jpeg")
            )
            .outputConfig(MultimodalRequest.OutputConfig.builder()
                .maxTokens(256)
                .temperature(0.7)
                .topP(0.9)
                .build())
            .build();
    }

    @Benchmark
    public void benchmarkSimpleInference(Blackhole bh) throws Exception {
        MultimodalResponse response = processor.process(singleRequest)
            .await().atMost(java.time.Duration.ofSeconds(30));
        bh.consume(response);
    }

    @Benchmark
    public void benchmarkDetailedInference(Blackhole bh) throws Exception {
        MultimodalResponse response = processor.process(detailedRequest)
            .await().atMost(java.time.Duration.ofSeconds(30));
        bh.consume(response);
    }

    @Benchmark
    @Threads(4)
    public void benchmarkConcurrentInference(Blackhole bh) throws Exception {
        MultimodalResponse response = processor.process(singleRequest)
            .await().atMost(java.time.Duration.ofSeconds(30));
        bh.consume(response);
    }

    @Benchmark
    @Threads(8)
    public void benchmarkHighConcurrencyInference(Blackhole bh) throws Exception {
        MultimodalResponse response = processor.process(singleRequest)
            .await().atMost(java.time.Duration.ofSeconds(30));
        bh.consume(response);
    }

    private byte[] createTestImage() {
        // Create a minimal valid JPEG for benchmarking
        String base64Image = "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAMCAgMCAgMDAwMEAwMEBQgFBQQEBQoHBwYIDAoMCwsKCwsNDhIQDQ4RDgsLEBYQERMUFRUVDA8XGBYUGBIUFRT/2wBDAQMEBAUEBQkFBQkUDQsNFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBT/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAn/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBEQCEAwEPwAAf/9k=";
        return java.util.Base64.getDecoder().decode(base64Image);
    }

    public static void main(String[] args) throws RunnerException {
        Options opt = new OptionsBuilder()
            .include(LlamaCppMultimodalBenchmark.class.getSimpleName())
            .forks(2)
            .warmupIterations(3)
            .measurementIterations(5)
            .build();

        new Runner(opt).run();
    }
}
