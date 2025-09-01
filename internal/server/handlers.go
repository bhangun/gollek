package server

import (
	"fmt"
	"net/http"
	"runtime"
	"time"

	"github.com/bhangun/wayang-inference/internal/llm"
	"github.com/bhangun/wayang-inference/internal/streaming"
	"github.com/bhangun/wayang-inference/pkg/types"

	"github.com/gin-gonic/gin"
	"github.com/sirupsen/logrus"
)

// Handler contains the HTTP handlers for the API
type Handler struct {
	pool          *llm.WorkerPool
	streamHandler *streaming.StreamingHandler
	logger        *logrus.Logger
	startTime     time.Time
	version       string
}

// NewHandler creates a new handler instance
func NewHandler(pool *llm.WorkerPool, logger *logrus.Logger, version string) *Handler {
	return &Handler{
		pool:          pool,
		streamHandler: streaming.NewStreamingHandler(pool, logger),
		logger:        logger,
		startTime:     time.Now(),
		version:       version,
	}
}

// CompletionHandler handles text completion requests
func (h *Handler) CompletionHandler(c *gin.Context) {
	var req types.CompletionRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		h.logger.WithError(err).Error("Invalid request body")
		c.JSON(http.StatusBadRequest, types.ErrorResponse{
			Error: types.ErrorDetail{
				Message: "Invalid request body: " + err.Error(),
				Type:    "invalid_request_error",
				Code:    "400",
			},
		})
		return
	}

	// Validate request
	if err := req.Validate(); err != nil {
		if apiErr, ok := err.(*types.APIError); ok {
			c.JSON(http.StatusBadRequest, types.ErrorResponse{
				Error: types.ErrorDetail{
					Message: apiErr.Message,
					Type:    apiErr.Type,
					Code:    apiErr.Code,
				},
			})
		} else {
			c.JSON(http.StatusBadRequest, types.ErrorResponse{
				Error: types.ErrorDetail{
					Message: err.Error(),
					Type:    "invalid_request_error",
					Code:    "400",
				},
			})
		}
		return
	}

	// Log request
	h.logger.WithFields(logrus.Fields{
		"prompt_length": len(req.Prompt),
		"max_tokens":    req.MaxTokens,
		"temperature":   req.Temperature,
		"stream":        req.Stream,
	}).Info("Processing completion request")

	// Handle streaming requests
	if streaming.IsStreamingRequest(&req) {
		h.streamHandler.HandleStreamRequest(c.Writer, c.Request, &req)
		return
	}

	// Handle non-streaming requests
	response, err := h.pool.Submit(c.Request.Context(), &req)
	if err != nil {
		h.logger.WithError(err).Error("Failed to process completion request")

		statusCode := http.StatusInternalServerError
		errorType := "server_error"

		// Handle specific error types
		if err.Error() == "request queue is full" {
			statusCode = http.StatusTooManyRequests
			errorType = "rate_limit_exceeded"
		}

		c.JSON(statusCode, types.ErrorResponse{
			Error: types.ErrorDetail{
				Message: err.Error(),
				Type:    errorType,
				Code:    http.StatusText(statusCode),
			},
		})
		return
	}

	c.JSON(http.StatusOK, response)
}

// HealthHandler handles health check requests
func (h *Handler) HealthHandler(c *gin.Context) {
	poolStats := h.pool.GetStats()
	engineStats := h.pool.GetEngineStats()

	status := "healthy"
	if !h.pool.IsHealthy() {
		status = "unhealthy"
	}

	// Get system info
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	health := types.HealthResponse{
		Status:    status,
		Timestamp: time.Now(),
		Version:   h.version,
		Model: types.HealthModelInfo{
			Path:      "", // Would need to get this from config
			Loaded:    engineStats.ModelLoaded,
			GPULayers: engineStats.GPULayersLoaded,
		},
		System: types.HealthSystemInfo{
			CPUCount:    runtime.NumCPU(),
			MemoryUsage: formatBytes(m.Alloc),
			Platform:    runtime.GOOS + "/" + runtime.GOARCH,
		},
		Stats: types.HealthStats{
			ActiveRequests: int(poolStats.ActiveRequests),
			QueuedRequests: int(poolStats.QueuedRequests),
			TotalRequests:  poolStats.TotalRequests,
			AverageLatency: 0, // Would need to track this
			Uptime:         time.Since(h.startTime),
		},
	}

	if status == "healthy" {
		c.JSON(http.StatusOK, health)
	} else {
		c.JSON(http.StatusServiceUnavailable, health)
	}
}

// ModelInfoHandler returns information about the loaded model
func (h *Handler) ModelInfoHandler(c *gin.Context) {
	if !h.pool.IsHealthy() {
		c.JSON(http.StatusServiceUnavailable, types.ErrorResponse{
			Error: types.ErrorDetail{
				Message: "Model not loaded",
				Type:    "model_not_loaded",
				Code:    "503",
			},
		})
		return
	}

	// Get model info from first engine (they should all be the same)
	var modelInfo *types.ModelInfo
	if len(h.pool.GetEngineStats()) > 0 {
		// For now, create a basic model info
		// In a full implementation, you'd get this from the engine
		modelInfo = &types.ModelInfo{
			Name:         "llama-model",
			Architecture: "llama",
			ContextSize:  2048, // Would get from actual model
		}
	}

	c.JSON(http.StatusOK, modelInfo)
}

// MetricsHandler returns detailed metrics
func (h *Handler) MetricsHandler(c *gin.Context) {
	poolStats := h.pool.GetStats()
	engineStats := h.pool.GetEngineStats()

	metrics := gin.H{
		"timestamp": time.Now(),
		"uptime":    time.Since(h.startTime),
		"pool": gin.H{
			"active_requests": poolStats.ActiveRequests,
			"queued_requests": poolStats.QueuedRequests,
			"total_requests":  poolStats.TotalRequests,
			"worker_count":    poolStats.WorkerCount,
			"queue_capacity":  poolStats.QueueCapacity,
			"queue_length":    poolStats.QueueLength,
		},
		"engine": gin.H{
			"model_loaded":       engineStats.ModelLoaded,
			"gpu_layers_loaded":  engineStats.GPULayersLoaded,
			"memory_usage":       engineStats.MemoryUsage,
			"tokens_processed":   engineStats.TokensProcessed,
			"requests_served":    engineStats.RequestsServed,
			"average_tokens_sec": engineStats.AverageTokensPerSec,
		},
	}

	c.JSON(http.StatusOK, metrics)
}

// formatBytes formats bytes to human readable string
func formatBytes(bytes uint64) string {
	const unit = 1024
	if bytes < unit {
		return fmt.Sprintf("%d B", bytes)
	}
	div, exp := int64(unit), 0
	for n := bytes / unit; n >= unit; n /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %cB", float64(bytes)/float64(div), "KMGTPE"[exp])
}
