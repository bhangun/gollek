package tech.kayys.gollek.runtime.plan;

import tech.kayys.gollek.ir.GValueId;
import java.util.*;

public final class MemoryPlanner {
    public static Map<GValueId, Integer> plan(
            Map<GValueId, LifetimeAnalyzer.Lifetime> lifetimes) {
        Map<GValueId, Integer> slotMap = new HashMap<>();
        List<LifetimeAnalyzer.Lifetime> active = new ArrayList<>();
        int nextSlot = 0;

        List<Map.Entry<GValueId, LifetimeAnalyzer.Lifetime>> entries = new ArrayList<>(lifetimes.entrySet());
        entries.sort(Comparator.comparingInt(e -> e.getValue().start));
        for (var entry : entries) {
            GValueId id = entry.getKey();
            var lt = entry.getValue();
            // free expired
            active.removeIf(a -> a.end < lt.start);
            int slot = active.size();
            slotMap.put(id, slot);
            active.add(lt);
            nextSlot = Math.max(nextSlot, active.size());
        }
        return slotMap;
    }
}