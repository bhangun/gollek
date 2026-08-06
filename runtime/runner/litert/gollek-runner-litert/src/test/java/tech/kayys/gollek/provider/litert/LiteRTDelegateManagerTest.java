package tech.kayys.gollek.provider.litert;
 
 import org.junit.jupiter.api.*;
 
 import static org.junit.jupiter.api.Assertions.*;
 
 /**
  * Tests for LiteRT 2.0 Delegate Manager.
  */
 @DisplayName("LiteRT Delegate Manager Tests")
 class LiteRTDelegateManagerTest {
 
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
 }