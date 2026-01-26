package tech.kayys.golek.mcp;

import java.security.NoSuchAlgorithmException;
import java.security.SecureRandom;
import java.security.spec.InvalidKeySpecException;
import java.util.Base64;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

import javax.crypto.SecretKeyFactory;
import javax.crypto.spec.PBEKeySpec;


public class ApiKeyManager {
    
    private final Map<String, HashedApiKey> keys = new ConcurrentHashMap<>();
    private final SecureRandom random = new SecureRandom();
    
    public String generateApiKey(String name, Set<String> permissions, long expiresAt) {
        byte[] keyBytes = new byte[32];
        random.nextBytes(keyBytes);
        
        String apiKey = "sk-" + Base64.getEncoder().encodeToString(keyBytes)
            .replace("+", "").replace("/", "").substring(0, 48);
        
        String hash = hashApiKey(apiKey);
        HashedApiKey hashedKey = new HashedApiKey(
            hash,
            name,
            permissions,
            System.currentTimeMillis(),
            expiresAt
        );
        
        keys.put(hash, hashedKey);
        return apiKey;
    }
    
    public boolean validateApiKey(String apiKey) {
        String hash = hashApiKey(apiKey);
        HashedApiKey hashedKey = keys.get(hash);
        
        if (hashedKey == null) {
            return false;
        }
        
        // Check expiration
        if (hashedKey.expiresAt() > 0 && 
            System.currentTimeMillis() > hashedKey.expiresAt()) {
            keys.remove(hash);
            return false;
        }
        
        return true;
    }
    
    public Set<String> getPermissions(String apiKey) {
        String hash = hashApiKey(apiKey);
        HashedApiKey hashedKey = keys.get(hash);
        return hashedKey != null ? hashedKey.permissions() : Set.of();
    }
    
    public void revokeApiKey(String apiKey) {
        String hash = hashApiKey(apiKey);
        keys.remove(hash);
    }
    
    public void rotateApiKey(String oldApiKey) {
        String oldHash = hashApiKey(oldApiKey);
        HashedApiKey oldKey = keys.get(oldHash);
        
        if (oldKey != null) {
            String newApiKey = generateApiKey(
                oldKey.name(),
                oldKey.permissions(),
                oldKey.expiresAt()
            );
            
            keys.remove(oldHash);
        }
    }
    
    private String hashApiKey(String apiKey) {
        try {
            byte[] salt = "llama-platform-salt".getBytes(); // In production, use unique salt
            PBEKeySpec spec = new PBEKeySpec(
                apiKey.toCharArray(),
                salt,
                10000,
                256
            );
            
            SecretKeyFactory factory = SecretKeyFactory.getInstance("PBKDF2WithHmacSHA256");
            byte[] hash = factory.generateSecret(spec).getEncoded();
            
            return Base64.getEncoder().encodeToString(hash);
        } catch (NoSuchAlgorithmException | InvalidKeySpecException e) {
            throw new RuntimeException("Failed to hash API key", e);
        }
    }
    
    public List<ApiKeyInfo> listApiKeys() {
        return keys.values().stream()
            .map(key -> new ApiKeyInfo(
                "***" + key.hash().substring(key.hash().length() - 8),
                key.name(),
                key.permissions(),
                key.createdAt(),
                key.expiresAt()
            ))
            .toList();
    }
    
    private record HashedApiKey(
        String hash,
        String name,
        Set<String> permissions,
        long createdAt,
        long expiresAt
    ) {}
    
    public record ApiKeyInfo(
        String keyPreview,
        String name,
        Set<String> permissions,
        long createdAt,
        long expiresAt
    ) {}
}
