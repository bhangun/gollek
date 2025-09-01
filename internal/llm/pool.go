package llm

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"github.com/bhangun/wayang-inference/internal/config"
	"github.com/bhangun/wayang-inference/pkg/types"

	"github.com/sirupsen/logrus"
)

// WorkerPool manages concurrent access to LLM engines
// Since llama.cpp is not thread-safe, we use a pool of workers to handle requests sequentially
type WorkerPool struct {
	engines     []Engine
	requestChan chan *WorkerRequest
	workers     []*Worker
	config      *config.LLMConfig
	logger      *logrus.Entry

	// Statistics
	activeRequests int64
	queuedRequests int64
	totalRequests  int64

	// Control
	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup
}

// WorkerRequest represents a request in the worker queue
type WorkerRequest struct {
	Request    *types.CompletionRequest
	Response   chan *WorkerResponse
	Context    context.Context
	IsStream   bool
	StreamChan chan StreamToken
}

// WorkerResponse represents a response from a worker
type WorkerResponse struct {
	Response *types.CompletionResponse
	Error    error
}

// Worker represents a single worker that processes requests
type Worker struct {
	ID     int
	Engine Engine
	logger *logrus.Entry
}

// NewWorkerPool creates a new worker pool
func NewWorkerPool(cfg *config.LLMConfig, logger *logrus.Logger) *WorkerPool {
	ctx, cancel := context.WithCancel(context.Background())

	pool := &WorkerPool{
		config:      cfg,
		requestChan: make(chan *WorkerRequest, cfg.MaxQueueSize),
		logger:      logger.WithField("component", "worker-pool"),
		ctx:         ctx,
		cancel:      cancel,
	}

	return pool
}

// Initialize creates and starts the worker pool
func (p *WorkerPool) Initialize() error {
	p.logger.WithFields(logrus.Fields{
		"workers":        p.config.WorkerPoolSize,
		"max_queue_size": p.config.MaxQueueSize,
	}).Info("Initializing worker pool")

	// Create engines for each worker
	p.engines = make([]Engine, p.config.WorkerPoolSize)
	p.workers = make([]*Worker, p.config.WorkerPoolSize)

	for i := 0; i < p.config.WorkerPoolSize; i++ {
		// Create engine instance
		engine := NewLlamaEngine(p.config, p.logger.Logger)
		if err := engine.Initialize(); err != nil {
			// Cleanup any already initialized engines
			for j := 0; j < i; j++ {
				p.engines[j].Close()
			}
			return fmt.Errorf("failed to initialize engine %d: %w", i, err)
		}

		p.engines[i] = engine
		p.workers[i] = &Worker{
			ID:     i,
			Engine: engine,
			logger: p.logger.WithField("worker_id", i),
		}
	}

	// Start workers
	for i, worker := range p.workers {
		p.wg.Add(1)
		go p.runWorker(worker, i)
	}

	p.logger.Info("Worker pool initialized successfully")
	return nil
}

// Submit submits a request to the worker pool
func (p *WorkerPool) Submit(ctx context.Context, req *types.CompletionRequest) (*types.CompletionResponse, error) {
	atomic.AddInt64(&p.totalRequests, 1)
	atomic.AddInt64(&p.queuedRequests, 1)
	defer atomic.AddInt64(&p.queuedRequests, -1)

	workerReq := &WorkerRequest{
		Request:  req,
		Response: make(chan *WorkerResponse, 1),
		Context:  ctx,
		IsStream: false,
	}

	// Check if we can queue the request
	select {
	case p.requestChan <- workerReq:
		// Request queued successfully
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-p.ctx.Done():
		return nil, fmt.Errorf("worker pool is shutting down")
	default:
		return nil, fmt.Errorf("request queue is full")
	}

	// Wait for response
	select {
	case resp := <-workerReq.Response:
		return resp.Response, resp.Error
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-p.ctx.Done():
		return nil, fmt.Errorf("worker pool is shutting down")
	}
}

// SubmitStream submits a streaming request to the worker pool
func (p *WorkerPool) SubmitStream(ctx context.Context, req *types.CompletionRequest) (<-chan StreamToken, error) {
	atomic.AddInt64(&p.totalRequests, 1)
	atomic.AddInt64(&p.queuedRequests, 1)
	defer atomic.AddInt64(&p.queuedRequests, -1)

	streamChan := make(chan StreamToken, 100)
	workerReq := &WorkerRequest{
		Request:    req,
		Context:    ctx,
		IsStream:   true,
		StreamChan: streamChan,
	}

	// Check if we can queue the request
	select {
	case p.requestChan <- workerReq:
		return streamChan, nil
	case <-ctx.Done():
		close(streamChan)
		return nil, ctx.Err()
	case <-p.ctx.Done():
		close(streamChan)
		return nil, fmt.Errorf("worker pool is shutting down")
	default:
		close(streamChan)
		return nil, fmt.Errorf("request queue is full")
	}
}

// GetStats returns pool statistics
func (p *WorkerPool) GetStats() *PoolStats {
	return &PoolStats{
		ActiveRequests: atomic.LoadInt64(&p.activeRequests),
		QueuedRequests: atomic.LoadInt64(&p.queuedRequests),
		TotalRequests:  atomic.LoadInt64(&p.totalRequests),
		WorkerCount:    len(p.workers),
		QueueCapacity:  cap(p.requestChan),
		QueueLength:    len(p.requestChan),
	}
}

// GetEngineStats returns aggregated engine statistics
func (p *WorkerPool) GetEngineStats() *EngineStats {
	if len(p.engines) == 0 {
		return &EngineStats{}
	}

	// Return stats from the first engine (they should be similar)
	return p.engines[0].GetStats()
}

// IsHealthy returns true if the pool is healthy
func (p *WorkerPool) IsHealthy() bool {
	if len(p.engines) == 0 {
		return false
	}

	// Check if at least one engine is healthy
	for _, engine := range p.engines {
		if engine.IsHealthy() {
			return true
		}
	}

	return false
}

// Close shuts down the worker pool
func (p *WorkerPool) Close() error {
	p.logger.Info("Shutting down worker pool")

	// Cancel context to signal workers to stop
	p.cancel()

	// Close request channel
	close(p.requestChan)

	// Wait for all workers to finish
	p.wg.Wait()

	// Close all engines
	for _, engine := range p.engines {
		if err := engine.Close(); err != nil {
			p.logger.WithError(err).Error("Error closing engine")
		}
	}

	p.logger.Info("Worker pool shut down complete")
	return nil
}

// runWorker runs a single worker
func (p *WorkerPool) runWorker(worker *Worker, workerIndex int) {
	defer p.wg.Done()

	worker.logger.Info("Worker started")
	defer worker.logger.Info("Worker stopped")

	for {
		select {
		case req, ok := <-p.requestChan:
			if !ok {
				// Channel closed, worker should exit
				return
			}

			p.processRequest(worker, req)

		case <-p.ctx.Done():
			// Pool is shutting down
			return
		}
	}
}

// processRequest processes a single request
func (p *WorkerPool) processRequest(worker *Worker, req *WorkerRequest) {
	atomic.AddInt64(&p.activeRequests, 1)
	defer atomic.AddInt64(&p.activeRequests, -1)

	startTime := time.Now()
	defer func() {
		duration := time.Since(startTime)
		worker.logger.WithFields(logrus.Fields{
			"duration":  duration,
			"is_stream": req.IsStream,
		}).Debug("Request processed")
	}()

	// Create a timeout context
	ctx, cancel := context.WithTimeout(req.Context, p.config.RequestTimeout)
	defer cancel()

	if req.IsStream {
		p.processStreamRequest(worker, req, ctx)
	} else {
		p.processNormalRequest(worker, req, ctx)
	}
}

// processNormalRequest processes a normal (non-streaming) request
func (p *WorkerPool) processNormalRequest(worker *Worker, req *WorkerRequest, ctx context.Context) {
	resp, err := worker.Engine.Generate(ctx, req.Request)

	workerResp := &WorkerResponse{
		Response: resp,
		Error:    err,
	}

	// Send response back
	select {
	case req.Response <- workerResp:
	case <-ctx.Done():
		// Context cancelled, no one is waiting for the response
	}
}

// processStreamRequest processes a streaming request
func (p *WorkerPool) processStreamRequest(worker *Worker, req *WorkerRequest, ctx context.Context) {
	defer close(req.StreamChan)

	tokenChan, err := worker.Engine.GenerateStream(ctx, req.Request)
	if err != nil {
		select {
		case req.StreamChan <- StreamToken{Error: err}:
		case <-ctx.Done():
		}
		return
	}

	// Forward tokens from engine to request stream
	for token := range tokenChan {
		select {
		case req.StreamChan <- token:
		case <-ctx.Done():
			return
		}

		// Stop if we received an error or completion
		if token.Error != nil || token.IsComplete {
			return
		}
	}
}

// PoolStats contains worker pool statistics
type PoolStats struct {
	ActiveRequests int64 `json:"active_requests"`
	QueuedRequests int64 `json:"queued_requests"`
	TotalRequests  int64 `json:"total_requests"`
	WorkerCount    int   `json:"worker_count"`
	QueueCapacity  int   `json:"queue_capacity"`
	QueueLength    int   `json:"queue_length"`
}
