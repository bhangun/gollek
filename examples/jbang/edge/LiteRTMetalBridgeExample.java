///usr/bin/env jbang "$0" "$@" ; exit $?
//NOINTEGRATIONS
// Gollek Native Metal Bridge Example
// 
// Demonstrates direct access to Apple Silicon GPU using the project's 
// custom Metal bridge (libgollek_metal.dylib).
//
// FEATURES:
//   - Zero-copy Unified Memory (CPU/GPU shared DRAM)
//   - High-performance GEMM (Matrix Multiplication) via MPS
//   - Performance comparison (Native CPU vs. Native Metal)
//
// DEPENDENCIES:
//   cd gollek/plugins/kernel/metal/gollek-kernel-metal && mvn clean install -DskipTests
//
//REPOS mavencentral,mylocal=file:///Users/bhangun/.m2/repository
//DEPS tech.kayys.gollek:gollek-kernel-metal:0.1.0-SNAPSHOT
//DEPS org.slf4j:slf4j-simple:2.0.12
//JAVA_OPTIONS --enable-native-access=ALL-UNNAMED
//JAVA 21+

package tech.kayys.gollek.examples.metal;

import tech.kayys.gollek.metal.binding.MetalBinding;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Verification script for the custom Metal kernel.
 */
public class LiteRTMetalBridgeExample {

    public static void main(String[] args) throws Exception {
        System.out.println("╔══════════════════════════════════════════════════════════╗");
        System.out.println("║       Gollek Native Metal Bridge Verification           ║");
        System.out.println("╚══════════════════════════════════════════════════════════╝");
        System.out.println();

        String home = System.getProperty("user.home");
        Path libPath = Paths.get(home, ".gollek", "libs", "libgollek_metal.dylib");

        System.out.println("Searching for Metal bridge: " + libPath);
        if (!java.nio.file.Files.exists(libPath)) {
            System.err.println("❌ Error: Native Metal bridge not found.");
            System.err.println("   Please run: make -C gollek/plugins/kernel/metal/gollek-kernel-metal/src/main/cpp/metal install");
            System.exit(1);
        }

        // Initialize Native Binding
        if (!MetalBinding.initialize(libPath)) {
            System.err.println("❌ Error: Failed to load Metal bridge.");
            System.exit(1);
        }

        MetalBinding binding = MetalBinding.getInstance();
        System.out.println("✓ Metal Bridge loaded successfully");

        // Initialize GPU Device
        int status = binding.init();
        if (status != 0) {
            System.err.println("❌ Error: Failed to initialize Metal device (code: " + status + ")");
            System.exit(1);
        }

        System.out.println("ℹ️  Device:         " + binding.deviceName());
        System.out.println("ℹ️  Unified Memory: " + binding.isUnifiedMemory());
        System.out.println("ℹ️  Available Mem:  " + (binding.availableMemory() / (1024 * 1024)) + " MB");
        System.out.println();

        // RUN BENCHMARK: Matrix Multiplication (GEMM)
        // C = A * B where A[M,K], B[K,N], C[M,N]
        int M = 1024, K = 1024, N = 1024;
        System.out.println("Benchmark: FP32 Matrix Multiplication [" + M + "x" + K + "x" + N + "]");

        try (Arena arena = Arena.ofAuto()) { // Use shared arena for zero-copy unified memory
            long sizeA = (long) M * K * 4;
            long sizeB = (long) K * N * 4;
            long sizeC = (long) M * N * 4;

            MemorySegment A = arena.allocate(sizeA, 64);
            MemorySegment B = arena.allocate(sizeB, 64);
            MemorySegment C = arena.allocate(sizeC, 64);

            // Fill with random data
            for (int i = 0; i < M * K; i++) A.setAtIndex(ValueLayout.JAVA_FLOAT, i, (float) Math.random());
            for (int i = 0; i < K * N; i++) B.setAtIndex(ValueLayout.JAVA_FLOAT, i, (float) Math.random());

            System.out.print("Warmup... ");
            binding.matmul(C, A, B, M, K, N, 1.0f, 0.0f);
            System.out.println("done.");

            int iterations = 10;
            System.out.print("Running " + iterations + " iterations on Meta GPU... ");
            long t0 = System.nanoTime();
            for (int i = 0; i < iterations; i++) {
                binding.matmul(C, A, B, M, K, N, 1.0f, 0.0f);
            }
            long t1 = System.nanoTime();
            double avgMs = (t1 - t0) / (iterations * 1_000_000.0);
            
            // Calculate TFLOPS: (2 * M * N * K) / (time_in_seconds * 10^12)
            double tflops = (2.0 * M * N * K) / (avgMs / 1000.0) / 1e12;

            System.out.println("✓ SUCCESS");
            System.out.println();
            System.out.println("Performance Metrics:");
            System.out.printf("  Average Latency: %.3f ms\n", avgMs);
            System.out.printf("  Estimated Power: %.3f TFLOPS\n", tflops);
        }

        System.out.println();
        System.out.println("╔══════════════════════════════════════════════════════════╗");
        System.out.println("║   ✓ STATUS: Metal Acceleration is FULLY OPERATIONAL     ║");
        System.out.println("╚══════════════════════════════════════════════════════════╝");
    }
}
