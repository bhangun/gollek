package tech.kayys.gollek.server.jobs;

import tech.kayys.gollek.sdk.model.PullProgress;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.Future;

public class JobRecord {
    private final String jobId;
    private volatile String status = "PENDING";
    private final List<PullProgress> history = Collections.synchronizedList(new ArrayList<>());
    private volatile String error;
    private volatile Future<?> future;

    public JobRecord(String jobId) {
        this.jobId = jobId;
    }

    public void addProgress(PullProgress p) {
        history.add(p);
    }

    public List<PullProgress> getHistory() {
        return history;
    }

    public String getJobId() {
        return jobId;
    }

    public void setStatus(String status) {
        this.status = status;
    }

    public String getStatus() {
        return status;
    }

    public boolean isFinished() {
        return "COMPLETED".equals(status) || "FAILED".equals(status) || "CANCELLED".equals(status);
    }

    public void setError(String error) {
        this.error = error;
    }

    public String getError() {
        return error;
    }

    public void setFuture(Future<?> future) {
        this.future = future;
    }

    public boolean cancel() {
        if (future != null) {
            boolean cancelled = future.cancel(true);
            if (cancelled) {
                setStatus("CANCELLED");
            }
            return cancelled;
        }
        return false;
    }

    public Info toInfo() {
        PullProgress last = history.isEmpty() ? null : history.get(history.size() - 1);
        return new Info(jobId, status, last, error);
    }

    public static record Info(String jobId, String status, PullProgress lastProgress, String error) { }
}
