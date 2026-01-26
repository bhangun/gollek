package tech.kayys.golek.mcp;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.LongAdder;


public class Metrics {
    
    private final Map<String, Counter> counters = new ConcurrentHashMap<>();
    private final Map<String, Histogram> histograms = new ConcurrentHashMap<>();
    private final Map<String, Gauge> gauges = new ConcurrentHashMap<>();
    
    public void incrementCounter(String name) {
        counters.computeIfAbsent(name, k -> new Counter()).increment();
    }
    
    public void incrementCounter(String name, long value) {
        counters.computeIfAbsent(name, k -> new Counter()).increment(value);
    }
    
    public void recordHistogram(String name, long value) {
        histograms.computeIfAbsent(name, k -> new Histogram()).record(value);
    }
    
    public void setGauge(String name, long value) {
        gauges.computeIfAbsent(name, k -> new Gauge()).set(value);
    }
    
    public long getCounter(String name) {
        Counter counter = counters.get(name);
        return counter != null ? counter.get() : 0;
    }
    
    public HistogramSnapshot getHistogram(String name) {
        Histogram histogram = histograms.get(name);
        return histogram != null ? histogram.snapshot() : null;
    }
    
    public long getGauge(String name) {
        Gauge gauge = gauges.get(name);
        return gauge != null ? gauge.get() : 0;
    }
    
    public Map<String, Object> getAllMetrics() {
        Map<String, Object> all = new ConcurrentHashMap<>();
        
        counters.forEach((name, counter) -> 
            all.put("counter." + name, counter.get()));
        
        histograms.forEach((name, histogram) -> 
            all.put("histogram." + name, histogram.snapshot()));
        
        gauges.forEach((name, gauge) -> 
            all.put("gauge." + name, gauge.get()));
        
        return all;
    }
    
    public void reset() {
        counters.clear();
        histograms.clear();
        gauges.clear();
    }
    
    private static class Counter {
        private final LongAdder value = new LongAdder();
        
        void increment() {
            value.increment();
        }
        
        void increment(long delta) {
            value.add(delta);
        }
        
        long get() {
            return value.sum();
        }
    }
    
    private static class Histogram {
        private final LongAdder count = new LongAdder();
        private final LongAdder sum = new LongAdder();
        private final AtomicLong min = new AtomicLong(Long.MAX_VALUE);
        private final AtomicLong max = new AtomicLong(Long.MIN_VALUE);
        
        void record(long value) {
            count.increment();
            sum.add(value);
            
            min.updateAndGet(current -> Math.min(current, value));
            max.updateAndGet(current -> Math.max(current, value));
        }
        
        HistogramSnapshot snapshot() {
            long c = count.sum();
            long s = sum.sum();
            return new HistogramSnapshot(
                c,
                s,
                c > 0 ? (double) s / c : 0.0,
                min.get(),
                max.get()
            );
        }
    }
    
    public record HistogramSnapshot(
        long count,
        long sum,
        double mean,
        long min,
        long max
    ) {}
    
    private static class Gauge {
        private final AtomicLong value = new AtomicLong(0);
        
        void set(long newValue) {
            value.set(newValue);
        }
        
        long get() {
            return value.get();
        }
    }
}
