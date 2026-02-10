package tech.kayys.wayang.memory.service;

import tech.kayys.wayang.memory.model.MemoryBackup;
import tech.kayys.wayang.memory.model.MemoryContext;
import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.time.Instant;
import java.util.List;
import java.util.zip.GZIPOutputStream;
import java.util.zip.GZIPInputStream;
import java.io.*;

@ApplicationScoped
public class MemoryBackupService {
    
    private static final Logger LOG = LoggerFactory.getLogger(MemoryBackupService.class);
    
    @Inject
    MemoryService memoryService;
    
    @Inject
    MemoryEventPublisher eventPublisher;

    public Uni<MemoryBackup> createBackup(String userId, List<String> sessionIds) {
        LOG.info("Creating backup for user: {} with {} sessions", userId, sessionIds.size());
        
        return Multi.createFrom().iterable(sessionIds)
            .onItem().transformToUniAndMerge(sessionId -> 
                memoryService.getContext(sessionId, userId))
            .collect().asList()
            .onItem().transformToUni(contexts -> 
                compressAndStore(userId, contexts))
            .onItem().invoke(backup -> 
                LOG.info("Backup created: {}", backup.getBackupId()));
    }

    public Uni<List<MemoryContext>> restoreBackup(String backupId) {
        LOG.info("Restoring backup: {}", backupId);
        
        return loadBackup(backupId)
            .onItem().transformToUni(backup -> 
                decompressBackup(backup))
            .onItem().transformToUni(contexts -> 
                Multi.createFrom().iterable(contexts)
                    .onItem().transformToUniAndMerge(context -> 
                        memoryService.storeContext(context))
                    .collect().asList()
                    .replaceWith(contexts))
            .onItem().invoke(contexts -> 
                LOG.info("Restored {} contexts from backup: {}", contexts.size(), backupId));
    }

    public Uni<Void> schedulePeriodicBackup(String userId, java.time.Duration interval) {
        LOG.info("Scheduling periodic backup for user: {} every: {}", userId, interval);
        
        // Implementation would use Quarkus Scheduler
        return eventPublisher.publishBackupScheduled(userId, interval);
    }

    private Uni<MemoryBackup> compressAndStore(String userId, List<MemoryContext> contexts) {
        return Uni.createFrom().item(() -> {
            try {
                ByteArrayOutputStream baos = new ByteArrayOutputStream();
                GZIPOutputStream gzipOut = new GZIPOutputStream(baos);
                ObjectOutputStream objectOut = new ObjectOutputStream(gzipOut);
                
                objectOut.writeObject(contexts);
                objectOut.close();
                
                byte[] compressedData = baos.toByteArray();
                
                MemoryBackup backup = new MemoryBackup(
                    java.util.UUID.randomUUID().toString(),
                    userId,
                    contexts.size(),
                    compressedData.length,
                    Instant.now(),
                    compressedData
                );
                
                // Store backup to persistent storage (S3, filesystem, etc.)
                storeBackupData(backup);
                
                return backup;
            } catch (IOException e) {
                throw new RuntimeException("Failed to compress backup", e);
            }
        });
    }

    private Uni<MemoryBackup> loadBackup(String backupId) {
        // Implementation would load from persistent storage
        return Uni.createFrom().nullItem();
    }

    private Uni<List<MemoryContext>> decompressBackup(MemoryBackup backup) {
        return Uni.createFrom().item(() -> {
            try {
                ByteArrayInputStream bais = new ByteArrayInputStream(backup.getData());
                GZIPInputStream gzipIn = new GZIPInputStream(bais);
                ObjectInputStream objectIn = new ObjectInputStream(gzipIn);
                
                @SuppressWarnings("unchecked")
                List<MemoryContext> contexts = (List<MemoryContext>) objectIn.readObject();
                objectIn.close();
                
                return contexts;
            } catch (IOException | ClassNotFoundException e) {
                throw new RuntimeException("Failed to decompress backup", e);
            }
        });
    }

    private void storeBackupData(MemoryBackup backup) {
        // Implementation would store to S3, filesystem, etc.
        LOG.info("Storing backup data for backup: {}", backup.getBackupId());
    }
}