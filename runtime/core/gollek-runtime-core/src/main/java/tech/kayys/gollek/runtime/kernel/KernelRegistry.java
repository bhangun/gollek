package tech.kayys.gollek.runtime.kernel;


import java.util.*;

public final class KernelRegistry {
    private final Map<String, List<KernelCandidate>> map = new HashMap<>();

    public void register(String opType, KernelCandidate candidate) {
        map.computeIfAbsent(opType, k -> new ArrayList<>())
                .add(candidate);
    }

    public List<KernelCandidate> get(String opType) {
        return map.getOrDefault(opType, List.of());
    }
}