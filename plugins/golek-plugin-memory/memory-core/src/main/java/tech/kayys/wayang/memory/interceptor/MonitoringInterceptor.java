package tech.kayys.wayang.memory.interceptor;

import tech.kayys.wayang.memory.metrics.MemoryMetrics;
import io.quarkus.arc.Arc;
import jakarta.interceptor.AroundInvoke;
import jakarta.interceptor.Interceptor;
import jakarta.interceptor.InvocationContext;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

@Monitored
@Interceptor
public class MonitoringInterceptor {
    
    private static final Logger LOG = LoggerFactory.getLogger(MonitoringInterceptor.class);

    @AroundInvoke
    public Object monitor(InvocationContext context) throws Exception {
        long startTime = System.currentTimeMillis();
        String methodName = context.getMethod().getName();
        
        try {
            Object result = context.proceed();
            long duration = System.currentTimeMillis() - startTime;
            
            LOG.debug("Method {} completed in {} ms", methodName, duration);
            
            // Record metrics
            MemoryMetrics metrics = Arc.container().instance(MemoryMetrics.class).get();
            if (methodName.contains("getContext")) {
                metrics.recordRetrieval();
                metrics.recordRetrievalTime(duration);
            } else if (methodName.contains("store")) {
                metrics.recordStore();
            }
            
            return result;
        } catch (Exception e) {
            LOG.error("Method {} failed after {} ms", 
                methodName, System.currentTimeMillis() - startTime, e);
            throw e;
        }
    }
}