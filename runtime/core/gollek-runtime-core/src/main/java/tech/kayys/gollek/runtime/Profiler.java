package tech.kayys.gollek.runtime;

public interface Profiler {
    void onOpStart(String opName);

    void onOpEnd(String opName, long nanos);
}