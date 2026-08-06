package tech.kayys.gollek.provider.litert;
 
 import org.junit.jupiter.api.Test;
 import org.junit.jupiter.api.io.TempDir;
 import tech.kayys.gollek.provider.core.session.EwmaAdaptiveSessionEvictionPolicy;
 
 import java.nio.file.Files;
 import java.nio.file.Path;
 import java.time.Duration;
 
 import static org.junit.jupiter.api.Assertions.assertNotNull;
 import static org.junit.jupiter.api.Assertions.assertTrue;
 import static org.junit.jupiter.api.Assertions.assertEquals;
 
 class LiteRTSessionManagerTest {
 
     @TempDir
     Path tempDir;
 
     @Test
     void adaptiveIdleTimeoutRespondsToPressureTelemetry() {
         LiteRTSessionManager manager = new LiteRTSessionManager(config(tempDir), new EwmaAdaptiveSessionEvictionPolicy());
 
         int baseline = manager.adaptiveIdleTimeoutSeconds();
         assertEquals(300, baseline);
 
         for (int i = 0; i < 8; i++) {
             manager.recordAdaptiveTelemetryForTest(true, 0);
         }
         int tightened = manager.adaptiveIdleTimeoutSeconds();
         assertTrue(tightened < baseline);
         assertTrue(manager.adaptivePressureScoreForTest() > 0.60d);
 
         for (int i = 0; i < 12; i++) {
             manager.recordAdaptiveTelemetryForTest(false, 0);
         }
         int recovered = manager.adaptiveIdleTimeoutSeconds();
         assertTrue(recovered > tightened);
         assertTrue(manager.adaptivePressureScoreForTest() < 0.35d);
     }
 
     @Test
     void acquiresAndReleasesSessionUsingPool() throws Exception {
         LiteRTSessionManager manager = new LiteRTSessionManager(config(tempDir), new EwmaAdaptiveSessionEvictionPolicy());
 
         Path modelPath = tempDir.resolve("demo.litertlm");
         Files.writeString(modelPath, "dummy");
         LiteRTRunnerConfig runnerConfig = new LiteRTRunnerConfig(1, false, false, "auto", "auto");
         DummyRunner runner = new DummyRunner();
 
         LiteRTSessionManager.SessionContext ctx = manager.getSession(
                 "tenant-a",
                 "demo",
                 modelPath,
                 runnerConfig,
                 (ignoredPath, ignoredConfig) -> runner);
         manager.releaseSession("tenant-a", "demo", ctx);
 
         LiteRTSessionManager.SessionContext reused = manager.getSession(
                 "tenant-a",
                 "demo",
                 modelPath,
                 runnerConfig,
                 (ignoredPath, ignoredConfig) -> runner);
         assertNotNull(reused);
 
         manager.releaseSession("tenant-a", "demo", reused);
         manager.shutdown();
         assertTrue(runner.closed > 0);
     }
 
     static final class DummyRunner extends LiteRTCpuRunner {
         int closed;
 
         @Override
         public boolean health() {
             return true;
         }
 
         @Override
         public void close() {
             closed++;
         }
     }
 
     private LiteRTProviderConfig config(Path basePath) {
         return new LiteRTProviderConfig() {
             @Override
             public boolean enabled() {
                 return true;
             }
 
             @Override
             public String modelBasePath() {
                 return basePath.toString();
             }
 
             @Override
             public int threads() {
                 return 1;
             }
 
             @Override
             public boolean gpuEnabled() {
                 return false;
             }
 
             @Override
             public boolean autoMetalEnabled() {
                 return false;
             }
 
             @Override
             public boolean npuEnabled() {
                 return false;
             }
 
             @Override
             public String gpuBackend() {
                 return "auto";
             }
 
             @Override
             public String npuType() {
                 return "auto";
             }
 
             @Override
             public Duration defaultTimeout() {
                 return Duration.ofSeconds(1);
             }
 
             @Override
             public SessionConfig session() {
                 return new SessionConfig() {
                     @Override
                     public int maxPerTenant() {
                         return 2;
                     }
 
                     @Override
                     public int idleTimeoutSeconds() {
                         return 300;
                     }
 
                     @Override
                     public int maxTotal() {
                         return 8;
                     }
                 };
             }
         };
     }
 }
