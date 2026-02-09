import io.quarkus.test.junit.QuarkusTest;
import org.junit.jupiter.api.*;
import static org.junit.jupiter.api.Assertions.*;

import java.nio.file.Path;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.UUID;

/**
* Integration tests for GGUF adapter
* Requires a small GGUF model file for testing
*/
@QuarkusTest
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class LlamaCppRunnerTest {

private static LlamaCppRunner runner;
private static ModelManifest testManifest;
private static TenantContext testTenant;

@BeforeAll
public static void setup() throws Exception {
// Setup test model (use a small model like TinyLlama)
testTenant = TenantContext.of(new TenantId("test-tenant"));

testManifest = ModelManifest.builder()
.modelId("test-gguf-model")
.name("Test GGUF Model")
.version("1.0")
.tenantId(testTenant.tenantId())
.artifacts(Map.of(
ModelFormat.GGUF,
ArtifactLocation.of("file:///tmp/test-model.gguf")
))
.supportedDevices(List.of(SupportedDevice.cpu()))
.resourceRequirements(ResourceRequirements.builder()
.minMemory(MemorySize.ofMegabytes(512))
.build())
.build();

runner = new LlamaCppRunner();
}

@Test
@Order(1)
public void testInitialization() throws Exception {
Map
<String , Object> config = Map.of(
"n_threads", 4,
"n_ctx", 512,
"n_gpu_layers", 0
);

assertDoesNotThrow(() ->
runner.initialize(testManifest, config, testTenant)
);

assertTrue(runner.health().isHealthy());
}

@Test
@Order(2)
public void testSimpleInference() {
InferenceRequest request = InferenceRequest.builder()
.requestId(UUID.randomUUID().toString())
.input("prompt", "Hello, my name is")
.parameter("max_tokens", 20)
.parameter("temperature", 0.7f)
.build();

RequestContext context = RequestContext.builder()
.tenantContext(testTenant)
.timeout(Duration.ofSeconds(30))
.build();

InferenceResponse response = runner.infer(request, context);

assertNotNull(response);
assertEquals(request.requestId(), response.requestId());
assertTrue(response.hasOutput("text"));
assertFalse(response.getOutput("text", String.class).isEmpty());

System.out.println("Generated: " + response.getOutput("text", String.class));
}

@Test
@Order(3)
public void testConcurrentInferences() throws Exception {
int concurrentRequests = 5;
List
<CompletableFuture
<InferenceResponse>> futures = new ArrayList
<>();

for (int i = 0; i
< concurrentRequests ; i++) {
    InferenceRequest request= InferenceRequest.builder()
    .requestId(UUID.randomUUID().toString())
    .input("prompt" , "Test prompt " + i)
    .parameter("max_tokens" , 10)
    .build();
    RequestContext context= RequestContext.builder()
    .tenantContext(testTenant)
    .timeout(Duration.ofSeconds(30))
    .build();
    CompletableFuture
<InferenceResponse> future =
runner.inferAsync(request, context).toCompletableFuture();

futures.add(future);
}

CompletableFuture.allOf(futures.toArray(new CompletableFuture[0])).get();

for (CompletableFuture
<InferenceResponse> future : futures) {
InferenceResponse response = future.get();
assertNotNull(response);
assertTrue(response.hasOutput("text"));
}
}

@Test
@Order(4)
public void testMetrics() {
ResourceMetrics metrics = runner.getMetrics();

assertNotNull(metrics);
assertTrue(metrics.getTotalRequests() > 0);
assertTrue(metrics.getMemoryUsedMb() > 0);
}

@Test
@Order(5)
public void testWarmup() {
assertDoesNotThrow(() -> runner.warmup(List.of()));
}

@AfterAll
public static void tearDown() {
if (runner != null) {
runner.close();
}
}
}
