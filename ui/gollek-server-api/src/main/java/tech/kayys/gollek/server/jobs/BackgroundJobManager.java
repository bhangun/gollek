package tech.kayys.gollek.server.jobs;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;

import io.smallrye.mutiny.Multi;
import tech.kayys.gollek.sdk.model.PullProgress;
import tech.kayys.gollek.server.SdkProvider;

import java.util.List;
import java.util.Optional;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.ArrayList;
import java.util.Collections;

/**
 * Simple background job manager for long-running tasks (model pulls).
 */
@ApplicationScoped
public class BackgroundJobManager {

    private final ExecutorService executor = Executors.newCachedThreadPool(r -> {
        Thread t = new Thread(r);
        t.setDaemon(true);
        return t;
    });

    private final ConcurrentMap<String, JobRecord> jobs = new ConcurrentHashMap<>();

    @Inject
    SdkProvider sdkProvider;

    public String startPullJob(String modelSpec, String revision, boolean force) {
        String jobId = UUID.randomUUID().toString();
        JobRecord jr = new JobRecord(jobId);
        jobs.put(jobId, jr);

        var future = executor.submit(() -> {
            try {
                jr.setStatus("RUNNING");
                var sdk = sdkProvider.getSdk();
                sdk.pullModel(modelSpec, revision, force, p -> {
                    jr.addProgress(p);
                });
                jr.setStatus("COMPLETED");
            } catch (Exception e) {
                jr.setStatus("FAILED");
                jr.setError(e.getMessage());
            }
        });
        jr.setFuture(future);

        return jobId;
    }

    public Multi<PullProgress> streamProgress(String jobId) {
        JobRecord jr = jobs.get(jobId);
        if (jr == null) {
            return Multi.createFrom().failure(new IllegalArgumentException("No such job: " + jobId));
        }

        return Multi.createFrom().emitter(emitter -> {
            List<PullProgress> history = jr.getHistory();
            int idx = 0;
            synchronized (history) {
                for (PullProgress p : history) {
                    emitter.emit(p);
                }
                idx = history.size();
            }

            while (!jr.isFinished()) {
                try {
                    Thread.sleep(200);
                } catch (InterruptedException ignored) {
                }
                synchronized (history) {
                    while (idx < history.size()) {
                        emitter.emit(history.get(idx++));
                    }
                }
            }

            synchronized (history) {
                while (idx < history.size()) {
                    emitter.emit(history.get(idx++));
                }
            }
            emitter.complete();
        });
    }

    public Optional<JobRecord.Info> getJobInfo(String jobId) {
        JobRecord jr = jobs.get(jobId);
        return jr == null ? Optional.empty() : Optional.of(jr.toInfo());
    }

    public java.util.List<JobRecord.Info> listJobs() {
        return jobs.values().stream().map(JobRecord::toInfo).toList();
    }

    public boolean cancelJob(String jobId) {
        JobRecord jr = jobs.get(jobId);
        if (jr == null) return false;
        boolean cancelled = jr.cancel();
        if (cancelled) {
            jr.setError("Cancelled by user");
        }
        return cancelled;
    }
}

