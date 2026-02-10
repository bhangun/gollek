package tech.kayys.wayang.memory.service;

import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import org.eclipse.microprofile.reactive.messaging.Channel;
import org.eclipse.microprofile.reactive.messaging.Emitter;
import java.time.Duration;

@ApplicationScoped
public class MemoryEventPublisherExtensions {
    
    @Channel("backup-events")
    Emitter<BackupEvent> backupEventEmitter;

    public Uni<Void> publishBackupScheduled(String userId, Duration interval) {
        BackupEvent event = new BackupEvent(
            "BACKUP_SCHEDULED",
            userId,
            interval.toString(),
            java.time.Instant.now()
        );
        
        return Uni.createFrom().completionStage(
            backupEventEmitter.send(event))
            .replaceWithVoid();
    }

    public Uni<Void> publishBackupCompleted(String backupId, int sessionCount) {
        BackupEvent event = new BackupEvent(
            "BACKUP_COMPLETED",
            backupId,
            String.valueOf(sessionCount),
            java.time.Instant.now()
        );
        
        return Uni.createFrom().completionStage(
            backupEventEmitter.send(event))
            .replaceWithVoid();
    }

    public static class BackupEvent {
        public final String eventType;
        public final String entityId;
        public final String details;
        public final java.time.Instant timestamp;

        public BackupEvent(String eventType, String entityId, String details, java.time.Instant timestamp) {
            this.eventType = eventType;
            this.entityId = entityId;
            this.details = details;
            this.timestamp = timestamp;
        }
    }
}