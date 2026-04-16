package tech.kayys.gollek.provider.litert;
 
 import org.junit.jupiter.api.*;
 import tech.kayys.gollek.provider.litert.LiteRTDelegateManager.GpuBackend;
 import tech.kayys.gollek.provider.litert.LiteRTDelegateManager.NpuType;
 
 import static org.junit.jupiter.api.Assertions.*;
 
 /**
  * Tests for LiteRT 2.0 Delegate Manager compatibility layer.
  */
 @DisplayName("LiteRT Delegate Manager Tests")
 class LiteRTDelegateManagerTest {
 
     private LiteRTDelegateManager delegateManager;
 
     @BeforeEach
     void setUp() {
         this.delegateManager = new LiteRTDelegateManager();
     }
 
     @Test
     @DisplayName("Delegate manager should initialize successfully with no-args constructor")
     void testInitialization() {
         assertNotNull(delegateManager);
     }
 
     @Test
     @DisplayName("Hardware accelerator resolution logic should be correct")
     void testResolveAccelerators() {
         // CPU only
         int cpuOnly = LiteRTDelegateManager.resolveAccelerators(true, false, false);
         assertEquals(LiteRTNativeBindings.kLiteRtHwAcceleratorCpu, cpuOnly);
 
         // GPU + NPU
         int combo = LiteRTDelegateManager.resolveAccelerators(false, true, true);
         assertEquals(LiteRTNativeBindings.kLiteRtHwAcceleratorGpu | LiteRTNativeBindings.kLiteRtHwAcceleratorNpu, combo);
 
         // All
         int all = LiteRTDelegateManager.resolveAccelerators(true, true, true);
         assertEquals(LiteRTNativeBindings.kLiteRtHwAcceleratorCpu | LiteRTNativeBindings.kLiteRtHwAcceleratorGpu | LiteRTNativeBindings.kLiteRtHwAcceleratorNpu, all);
     }
 
     @Test
     @DisplayName("Deprecated methods should not crash")
     void testDeprecatedCompatibilityMethods() {
         assertDoesNotThrow(() -> {
             delegateManager.autoDetectAndInitializeDelegates();
             delegateManager.tryInitializeGpuDelegate(GpuBackend.METAL, "Metal");
             delegateManager.tryInitializeNpuDelegate(NpuType.HEXAGON, "Hexagon");
             delegateManager.getBestAvailableDelegate();
             delegateManager.cleanup();
         });
     }
 }