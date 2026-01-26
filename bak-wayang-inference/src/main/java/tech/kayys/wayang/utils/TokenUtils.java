package tech.kayys.golek.utils;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

import tech.kayys.golek.engine.LlamaCppBinding;

public class TokenUtils {
    
    public static String safeTokenToString(LlamaCppBinding binding, Arena arena, 
                                         MemorySegment model, int token) {
        try {
            String result = binding.tokenToString(arena, model, token);
            if (result == null || result.isEmpty()) {
                return "token_" + token;
            }
            // Clean up the string for display
            return result.replace("\n", "\\n")
                        .replace("\r", "\\r")
                        .replace("\t", "\\t")
                        .replace("\0", "\\0");
        } catch (Throwable t) {
            return "token_" + token;
        }
    }
    
    public static String safeTokenToString(LlamaCppBinding binding, Arena arena,
                                         MemorySegment model, int token, String fallback) {
        try {
            String result = binding.tokenToString(arena, model, token);
            return (result == null || result.isEmpty()) ? fallback : result;
        } catch (Throwable t) {
            return fallback;
        }
    }
}