package server

import (
	"context"
	"fmt"
	"net/http"
	"time"

	"github.com/bhangun/wayang-inference/internal/config"
	"github.com/bhangun/wayang-inference/internal/llm"

	"github.com/gin-gonic/gin"
	"github.com/sirupsen/logrus"
)

// Server represents the HTTP server
type Server struct {
	config     *config.Config
	pool       *llm.WorkerPool
	httpServer *http.Server
	logger     *logrus.Logger
	handler    *Handler
}

// New creates a new server instance
func New(cfg *config.Config, logger *logrus.Logger) (*Server, error) {
	// Create worker pool
	pool := llm.NewWorkerPool(&cfg.LLM, logger)
	if err := pool.Initialize(); err != nil {
		return nil, fmt.Errorf("failed to initialize worker pool: %w", err)
	}

	// Create handler
	handler := NewHandler(pool, logger, "1.0.0")

	// Setup Gin mode
	if cfg.Logs.Level == "debug" {
		gin.SetMode(gin.DebugMode)
	} else {
		gin.SetMode(gin.ReleaseMode)
	}

	// Create Gin router
	router := setupRoutes(handler, logger)

	// Create HTTP server
	httpServer := &http.Server{
		Addr:           cfg.GetServerAddress(),
		Handler:        router,
		ReadTimeout:    cfg.Server.ReadTimeout,
		WriteTimeout:   cfg.Server.WriteTimeout,
		MaxHeaderBytes: int(cfg.Server.MaxRequestSize),
	}

	return &Server{
		config:     cfg,
		pool:       pool,
		httpServer: httpServer,
		logger:     logger,
		handler:    handler,
	}, nil
}

// Start starts the HTTP server
func (s *Server) Start() error {
	s.logger.WithFields(logrus.Fields{
		"address":    s.config.GetServerAddress(),
		"gpu_layers": s.config.LLM.GPULayers,
		"workers":    s.config.LLM.WorkerPoolSize,
		"model_path": s.config.LLM.ModelPath,
	}).Info("Starting LLM inference server")

	return s.httpServer.ListenAndServe()
}

// Stop gracefully stops the server
func (s *Server) Stop(ctx context.Context) error {
	s.logger.Info("Shutting down server...")

	// Shutdown HTTP server
	if err := s.httpServer.Shutdown(ctx); err != nil {
		s.logger.WithError(err).Error("HTTP server shutdown error")
	}

	// Close worker pool
	if err := s.pool.Close(); err != nil {
		s.logger.WithError(err).Error("Worker pool shutdown error")
		return err
	}

	s.logger.Info("Server shutdown complete")
	return nil
}

// setupRoutes sets up the HTTP routes
func setupRoutes(handler *Handler, logger *logrus.Logger) *gin.Engine {
	router := gin.New()

	// Add middleware
	router.Use(gin.Recovery())
	router.Use(loggingMiddleware(logger))
	router.Use(corsMiddleware())
	router.Use(rateLimitMiddleware())

	// Health endpoints
	router.GET("/health", handler.HealthHandler)
	router.GET("/ready", handler.HealthHandler) // Kubernetes readiness probe
	router.GET("/live", handler.HealthHandler)  // Kubernetes liveness probe

	// API endpoints
	v1 := router.Group("/v1")
	{
		// OpenAI-compatible endpoints
		v1.POST("/completions", handler.CompletionHandler)
		v1.POST("/chat/completions", handler.CompletionHandler) // Alias for compatibility

		// Model information
		v1.GET("/models", handler.ModelInfoHandler)
		v1.GET("/model", handler.ModelInfoHandler)

		// Metrics
		v1.GET("/metrics", handler.MetricsHandler)
	}

	// Admin endpoints
	admin := router.Group("/admin")
	{
		admin.GET("/health", handler.HealthHandler)
		admin.GET("/metrics", handler.MetricsHandler)
	}

	return router
}

// loggingMiddleware adds structured logging to requests
func loggingMiddleware(logger *logrus.Logger) gin.HandlerFunc {
	return func(c *gin.Context) {
		start := time.Now()
		path := c.Request.URL.Path
		raw := c.Request.URL.RawQuery

		// Process request
		c.Next()

		// Calculate request processing time
		latency := time.Since(start)

		// Get status code
		statusCode := c.Writer.Status()

		// Build log fields
		fields := logrus.Fields{
			"status":     statusCode,
			"method":     c.Request.Method,
			"path":       path,
			"ip":         c.ClientIP(),
			"latency":    latency,
			"user_agent": c.Request.UserAgent(),
		}

		if raw != "" {
			fields["raw_query"] = raw
		}

		// Log based on status code
		if statusCode >= 500 {
			logger.WithFields(fields).Error("Server error")
		} else if statusCode >= 400 {
			logger.WithFields(fields).Warn("Client error")
		} else {
			logger.WithFields(fields).Info("Request processed")
		}
	}
}

// corsMiddleware adds CORS headers
func corsMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		c.Header("Access-Control-Allow-Origin", "*")
		c.Header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
		c.Header("Access-Control-Allow-Headers", "Origin, Content-Type, Content-Length, Accept-Encoding, X-CSRF-Token, Authorization")
		c.Header("Access-Control-Expose-Headers", "Content-Length")
		c.Header("Access-Control-Allow-Credentials", "true")

		if c.Request.Method == "OPTIONS" {
			c.AbortWithStatus(http.StatusNoContent)
			return
		}

		c.Next()
	}
}

// rateLimitMiddleware adds basic rate limiting
func rateLimitMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		// Basic implementation - in production, you'd want a more sophisticated rate limiter
		// like github.com/ulule/limiter with Redis backend

		// For now, just pass through
		c.Next()
	}
}
