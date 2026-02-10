package tech.kayys.wayang.memory.model;

import java.time.Instant;

public class MemoryBackup {
    private final String backupId;
    private final String userId;
    private final int sessionCount;
    private final long size;
    private final Instant createdAt;
    private final byte[] data;

    public MemoryBackup(
            String backupId,
            String userId,
            int sessionCount,
            long size,
            Instant createdAt,
            byte[] data) {
        this.backupId = backupId;
        this.userId = userId;
        this.sessionCount = sessionCount;
        this.size = size;
        this.createdAt = createdAt;
        this.data = data;
    }

    public String getBackupId() { return backupId; }
    public String getUserId() { return userId; }
    public int getSessionCount() { return sessionCount; }
    public long getSize() { return size; }
    public Instant getCreatedAt() { return createdAt; }
    public byte[] getData() { return data; }
}