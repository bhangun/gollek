package tech.kayys.golek.core.observability;

import io.opentelemetry.api.common.Attributes;

import java.time.Instant;
import java.util.Queue;
import java.util.concurrent.ConcurrentLinkedQueue;

/**
 * Aggregates log entries for centralized processing
 */
public class LogAggregator {
    private final Queue<LogEntry> logEntries = new ConcurrentLinkedQueue<>();
    private final int maxEntries;

    public LogAggregator() {
        this(1000); // Default max of 1000 entries
    }

    public LogAggregator(int maxEntries) {
        this.maxEntries = maxEntries;
    }

    /**
     * Add a log entry
     */
    public void addLog(String level, String message, Attributes attributes) {
        LogEntry entry = new LogEntry(Instant.now(), level, message, attributes);
        logEntries.offer(entry);

        // Trim if necessary
        while (logEntries.size() > maxEntries) {
            logEntries.poll();
        }
    }

    /**
     * Get recent log entries
     */
    public Queue<LogEntry> getRecentLogs() {
        return new ConcurrentLinkedQueue<>(logEntries);
    }

    /**
     * Clear all log entries
     */
    public void clearLogs() {
        logEntries.clear();
    }

    /**
     * Get the number of log entries
     */
    public int getLogCount() {
        return logEntries.size();
    }

    /**
     * Log entry record
     */
    public record LogEntry(
            Instant timestamp,
            String level,
            String message,
            Attributes attributes) {
    }
}