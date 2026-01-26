package tech.kayys.golek.mcp;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;

public class DistributedTracing {
    
    private static final ThreadLocal<TraceContext> currentContext = new ThreadLocal<>();
    private static final Map<String, Trace> traces = new ConcurrentHashMap<>();
    
    public static TraceContext startTrace(String operationName) {
        String traceId = UUID.randomUUID().toString();
        String spanId = UUID.randomUUID().toString();
        
        TraceContext context = new TraceContext(traceId, spanId, null, operationName);
        currentContext.set(context);
        
        Trace trace = new Trace(traceId);
        trace.addSpan(new Span(spanId, null, operationName, System.currentTimeMillis()));
        traces.put(traceId, trace);
        
        return context;
    }
    
    public static TraceContext startSpan(String operationName) {
        TraceContext parent = currentContext.get();
        if (parent == null) {
            return startTrace(operationName);
        }
        
        String spanId = UUID.randomUUID().toString();
        TraceContext context = new TraceContext(
            parent.traceId(),
            spanId,
            parent.spanId(),
            operationName
        );
        
        currentContext.set(context);
        
        Trace trace = traces.get(parent.traceId());
        if (trace != null) {
            trace.addSpan(new Span(spanId, parent.spanId(), operationName, System.currentTimeMillis()));
        }
        
        return context;
    }
    
    public static void endSpan() {
        TraceContext context = currentContext.get();
        if (context != null) {
            Trace trace = traces.get(context.traceId());
            if (trace != null) {
                trace.endSpan(context.spanId(), System.currentTimeMillis());
            }
            
            if (context.parentSpanId() != null) {
                // Restore parent context
                // (simplified - in real implementation, maintain context stack)
            } else {
                currentContext.remove();
            }
        }
    }
    
    public static void addTag(String key, String value) {
        TraceContext context = currentContext.get();
        if (context != null) {
            Trace trace = traces.get(context.traceId());
            if (trace != null) {
                trace.addTag(context.spanId(), key, value);
            }
        }
    }
    
    public static TraceContext getCurrentContext() {
        return currentContext.get();
    }
    
    public static Trace getTrace(String traceId) {
        return traces.get(traceId);
    }
    
    public static void clearOldTraces(long maxAgeMs) {
        long now = System.currentTimeMillis();
        traces.entrySet().removeIf(entry -> 
            now - entry.getValue().startTime > maxAgeMs);
    }
    
    public record TraceContext(
        String traceId,
        String spanId,
        String parentSpanId,
        String operationName
    ) {}
    
    public static class Trace {
        private final String traceId;
        private final long startTime;
        private final List<Span> spans = new ArrayList<>();
        
        public Trace(String traceId) {
            this.traceId = traceId;
            this.startTime = System.currentTimeMillis();
        }
        
        synchronized void addSpan(Span span) {
            spans.add(span);
        }
        
        synchronized void endSpan(String spanId, long endTime) {
            for (Span span : spans) {
                if (span.spanId.equals(spanId)) {
                    span.endTime = endTime;
                    break;
                }
            }
        }
        
        synchronized void addTag(String spanId, String key, String value) {
            for (Span span : spans) {
                if (span.spanId.equals(spanId)) {
                    span.tags.put(key, value);
                    break;
                }
            }
        }
        
        public String getTraceId() { return traceId; }
        public long getStartTime() { return startTime; }
        public List<Span> getSpans() { return new ArrayList<>(spans); }
        public long getDuration() {
            return spans.stream()
                .mapToLong(s -> s.endTime > 0 ? s.endTime - s.startTime : 0)
                .sum();
        }
    }
    
    public static class Span {
        private final String spanId;
        private final String parentSpanId;
        private final String operationName;
        private final long startTime;
        private long endTime;
        private final Map<String, String> tags = new ConcurrentHashMap<>();
        
        public Span(String spanId, String parentSpanId, String operationName, long startTime) {
            this.spanId = spanId;
            this.parentSpanId = parentSpanId;
            this.operationName = operationName;
            this.startTime = startTime;
        }
        
        public String getSpanId() { return spanId; }
        public String getParentSpanId() { return parentSpanId; }
        public String getOperationName() { return operationName; }
        public long getStartTime() { return startTime; }
        public long getEndTime() { return endTime; }
        public long getDuration() { return endTime > 0 ? endTime - startTime : 0; }
        public Map<String, String> getTags() { return new HashMap<>(tags); }
    }
}
