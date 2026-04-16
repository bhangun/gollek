/*
 * Test compilation of improved FlacDecoder
 */
package tech.kayys.gollek.safetensor.audio.processing;

import tech.kayys.suling.FlacLibraryCheck;

/**
 * Simple test to verify FlacDecoder integration compiles.
 */
public class FlacDecoderTest {
    
    public static void main(String[] args) {
        System.out.println("Testing FlacDecoder integration...");
        
        // Test library check
        boolean available = FlacLibraryCheck.isAvailable();
        System.out.println("Suling library available: " + available);
        
        if (available) {
            System.out.println("Version: " + FlacLibraryCheck.getVersion());
            System.out.println("Load source: " + FlacLibraryCheck.getLoadSource());
        } else {
            System.out.println("Diagnostics:");
            System.out.println(FlacLibraryCheck.getDiagnostics());
        }
        
        // Test decoder instantiation
        try {
            FlacDecoder decoder = new FlacDecoder();
            System.out.println("FlacDecoder instantiated successfully");
            System.out.println("Format: " + decoder.getFormat());
            System.out.println("Supports FLAC: " + decoder.supports("flac"));
        } catch (Exception e) {
            System.err.println("Error creating FlacDecoder: " + e.getMessage());
            e.printStackTrace();
        }
        
        System.out.println("Test completed successfully!");
    }
}
