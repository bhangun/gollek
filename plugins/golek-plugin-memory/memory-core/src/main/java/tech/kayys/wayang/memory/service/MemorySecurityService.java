package tech.kayys.wayang.memory.service;

import tech.kayys.wayang.memory.model.MemoryContext;
import tech.kayys.wayang.memory.model.SecurityScanResult;
import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.crypto.Cipher;
import javax.crypto.KeyGenerator;
import javax.crypto.SecretKey;
import javax.crypto.spec.SecretKeySpec;
import java.nio.charset.StandardCharsets;
import java.security.SecureRandom;
import java.util.Base64;
import java.util.List;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

@ApplicationScoped
public class MemorySecurityService {
    
    private static final Logger LOG = LoggerFactory.getLogger(MemorySecurityService.class);
    
    // PII Detection patterns
    private static final Pattern EMAIL_PATTERN = Pattern.compile(
        "\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b");
    private static final Pattern PHONE_PATTERN = Pattern.compile(
        "\\b\\d{3}[-.\\s]?\\d{3}[-.\\s]?\\d{4}\\b");
    private static final Pattern SSN_PATTERN = Pattern.compile(
        "\\b\\d{3}-\\d{2}-\\d{4}\\b");
    private static final Pattern CREDIT_CARD_PATTERN = Pattern.compile(
        "\\b\\d{4}[\\s-]?\\d{4}[\\s-]?\\d{4}[\\s-]?\\d{4}\\b");

    public Uni<SecurityScanResult> scanMemoryForPII(MemoryContext context) {
        LOG.info("Scanning memory for PII in session: {}", context.getSessionId());
        
        return Uni.createFrom().item(() -> {
            SecurityScanResult.Builder resultBuilder = SecurityScanResult.builder()
                .sessionId(context.getSessionId());
            
            context.getConversations().forEach(memory -> {
                String content = memory.getContent();
                
                if (EMAIL_PATTERN.matcher(content).find()) {
                    resultBuilder.addViolation("EMAIL_DETECTED", memory.getId(), "Email address found");
                }
                
                if (PHONE_PATTERN.matcher(content).find()) {
                    resultBuilder.addViolation("PHONE_DETECTED", memory.getId(), "Phone number found");
                }
                
                if (SSN_PATTERN.matcher(content).find()) {
                    resultBuilder.addViolation("SSN_DETECTED", memory.getId(), "Social Security Number found");
                }
                
                if (CREDIT_CARD_PATTERN.matcher(content).find()) {
                    resultBuilder.addViolation("CREDIT_CARD_DETECTED", memory.getId(), "Credit card number found");
                }
            });
            
            return resultBuilder.build();
        });
    }

    public Uni<MemoryContext> sanitizeMemory(MemoryContext context) {
        LOG.info("Sanitizing memory for session: {}", context.getSessionId());
        
        return Uni.createFrom().item(() -> {
            List<ConversationMemory> sanitizedMemories = context.getConversations().stream()
                .map(this::sanitizeConversationMemory)
                .collect(Collectors.toList());
            
            return new MemoryContext(
                context.getSessionId(),
                context.getUserId(),
                sanitizedMemories,
                context.getMetadata(),
                context.getCreatedAt(),
                context.getUpdatedAt()
            );
        });
    }

    public Uni<String> encryptSensitiveData(String data, String sessionId) {
        return Uni.createFrom().item(() -> {
            try {
                SecretKey key = generateSessionKey(sessionId);
                Cipher cipher = Cipher.getInstance("AES");
                cipher.init(Cipher.ENCRYPT_MODE, key);
                
                byte[] encrypted = cipher.doFinal(data.getBytes(StandardCharsets.UTF_8));
                return Base64.getEncoder().encodeToString(encrypted);
            } catch (Exception e) {
                LOG.error("Failed to encrypt data", e);
                throw new RuntimeException("Encryption failed", e);
            }
        });
    }

    public Uni<String> decryptSensitiveData(String encryptedData, String sessionId) {
        return Uni.createFrom().item(() -> {
            try {
                SecretKey key = generateSessionKey(sessionId);
                Cipher cipher = Cipher.getInstance("AES");
                cipher.init(Cipher.DECRYPT_MODE, key);
                
                byte[] decrypted = cipher.doFinal(Base64.getDecoder().decode(encryptedData));
                return new String(decrypted, StandardCharsets.UTF_8);
            } catch (Exception e) {
                LOG.error("Failed to decrypt data", e);
                throw new RuntimeException("Decryption failed", e);
            }
        });
    }

    private ConversationMemory sanitizeConversationMemory(ConversationMemory memory) {
        String sanitizedContent = memory.getContent();
        
        // Replace PII with placeholders
        sanitizedContent = EMAIL_PATTERN.matcher(sanitizedContent).replaceAll("[EMAIL_REDACTED]");
        sanitizedContent = PHONE_PATTERN.matcher(sanitizedContent).replaceAll("[PHONE_REDACTED]");
        sanitizedContent = SSN_PATTERN.matcher(sanitizedContent).replaceAll("[SSN_REDACTED]");
        sanitizedContent = CREDIT_CARD_PATTERN.matcher(sanitizedContent).replaceAll("[CARD_REDACTED]");
        
        return new ConversationMemory(
            memory.getId(),
            memory.getRole(),
            sanitizedContent,
            memory.getMetadata(),
            memory.getEmbedding(),
            memory.getTimestamp(),
            memory.getRelevanceScore()
        );
    }

    private SecretKey generateSessionKey(String sessionId) {
        // In production, use a proper key management system
        byte[] keyBytes = sessionId.getBytes(StandardCharsets.UTF_8);
        if (keyBytes.length < 16) {
            byte[] paddedKey = new byte[16];
            System.arraycopy(keyBytes, 0, paddedKey, 0, keyBytes.length);
            keyBytes = paddedKey;
        } else if (keyBytes.length > 16) {
            byte[] truncatedKey = new byte[16];
            System.arraycopy(keyBytes, 0, truncatedKey, 0, 16);
            keyBytes = truncatedKey;
        }
        
        return new SecretKeySpec(keyBytes, "AES");
    }
}