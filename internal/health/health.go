package health

import (
	"context"
	"fmt"
	"runtime"
	"time"

	"github.com/bhangun/wayang-inference/internal/llm"
	"github.com/bhangun/wayang-inference/pkg/types"

	"github.com/sirupsen/logrus"
)

// Checker performs health checks on the system
type Checker struct {
	pool      *llm.WorkerPool
	logger    *logrus.Entry
	startTime time.Time
	version   string
}

// NewChecker creates a new health checker
func NewChecker(pool *llm.WorkerPool, logger *logrus.Logger, version string) *Checker {
	return &Checker{
		pool:      pool,
		logger:    logger.WithField("component", "health-checker"),
		startTime: time.Now(),
		version:   version,
	}
}

// CheckHealth performs a comprehensive health check
func (c *Checker) CheckHealth(ctx context.Context) *types.HealthResponse {
	poolStats := c.pool.GetStats()
	engineStats := c.pool.GetEngineStats()

	status := "healthy"
	if !c.pool.IsHealthy() {
		status = "unhealthy"
		c.logger.Warn("Health check failed: pool is not healthy")
	}

	// Get system information
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	return &types.HealthResponse{
		Status:    status,
		Timestamp: time.Now(),
		Version:   c.version,
		Model: types.HealthModelInfo{
			Path:      "", // Would get from config
			Loaded:    engineStats.ModelLoaded,
			GPULayers: engineStats.GPULayersLoaded,
		},
		System: types.HealthSystemInfo{
			CPUCount:    runtime.NumCPU(),
			GPUCount:    c.getGPUCount(),
			MemoryUsage: c.formatBytes(m.Alloc),
			Platform:    runtime.GOOS + "/" + runtime.GOARCH,
		},
		Stats: types.HealthStats{
			ActiveRequests: int(poolStats.ActiveRequests),
			QueuedRequests: int(poolStats.QueuedRequests),
			TotalRequests:  poolStats.TotalRequests,
			AverageLatency: c.calculateAverageLatency(),
			Uptime:         time.Since(c.startTime),
		},
	}
}

// IsReady checks if the service is ready to accept requests
func (c *Checker) IsReady() bool {
	return c.pool.IsHealthy()
}

// IsLive checks if the service is alive (basic liveness check)
func (c *Checker) IsLive() bool {
	// Basic liveness check - service is alive if it can respond
	return true
}

// getGPUCount returns the number of available GPUs
func (c *Checker) getGPUCount() int {
	// This is a simplified implementation
	// In a real implementation, you'd query CUDA or other GPU APIs
	engineStats := c.pool.GetEngineStats()
	if engineStats.GPULayersLoaded > 0 {
		return 1 // Assume at least 1 GPU if layers are loaded
	}
	return 0
}

// formatBytes formats bytes to human readable format
func (c *Checker) formatBytes(bytes uint64) string {
	const unit = 1024
	if bytes < unit {
		return "0 MB"
	}

	units := []string{"B", "KB", "MB", "GB", "TB"}
	div := uint64(1)
	unitIndex := 0

	for bytes >= unit*div && unitIndex < len(units)-1 {
		div *= unit
		unitIndex++
	}

	return fmt.Sprintf("%.1f %s", float64(bytes)/float64(div), units[unitIndex])
}

// calculateAverageLatency calculates average request latency
func (c *Checker) calculateAverageLatency() time.Duration {
	// This is a placeholder - in a real implementation, you'd track request latencies
	return 0
}
