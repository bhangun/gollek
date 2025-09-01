package streaming

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/bhangun/wayang-inference/internal/llm"
	"github.com/bhangun/wayang-inference/pkg/types"
	"github.com/sirupsen/logrus"
)

// SSEWriter handles Server-Sent Events streaming
type SSEWriter struct {
	writer    http.ResponseWriter
	flusher   http.Flusher
	logger    *logrus.Entry
	requestID string
	modelName string
}

// NewSSEWriter creates a new SSE writer
func NewSSEWriter(w http.ResponseWriter, requestID, modelName string, logger *logrus.Logger) (*SSEWriter, error) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		return nil, fmt.Errorf("streaming not supported")
	}

	// Set SSE headers
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Headers", "Cache-Control")

	return &SSEWriter{
		writer:    w,
		flusher:   flusher,
		logger:    logger.WithField("component", "sse-writer"),
		requestID: requestID,
		modelName: modelName,
	}, nil
}

// StreamCompletion streams a completion response using Server-Sent Events
func (s *SSEWriter) StreamCompletion(ctx context.Context, tokenChan <-chan llm.StreamToken) error {
	s.logger.WithField("request_id", s.requestID).Info("Starting stream")

	created := time.Now().Unix()
	index := 0

	for {
		select {
		case <-ctx.Done():
			s.logger.WithField("request_id", s.requestID).Info("Stream cancelled by client")
			return ctx.Err()

		case token, ok := <-tokenChan:
			if !ok {
				// Channel closed, send final done message
				if err := s.writeDone(); err != nil {
					return err
				}
				s.logger.WithField("request_id", s.requestID).Info("Stream completed")
				return nil
			}

			if token.Error != nil {
				s.logger.WithError(token.Error).Error("Error in token stream")
				if err := s.writeError(token.Error); err != nil {
					return err
				}
				return token.Error
			}

			if token.IsComplete {
				// Generation completed normally
				if err := s.writeCompletion(created, index, token.FinishReason); err != nil {
					return err
				}
				if err := s.writeDone(); err != nil {
					return err
				}
				s.logger.WithField("request_id", s.requestID).Info("Stream completed")
				return nil
			}

			// Write token to stream
			if err := s.writeToken(created, index, token.Token); err != nil {
				return err
			}
		}
	}
}

// writeToken writes a single token to the stream
func (s *SSEWriter) writeToken(created int64, index int, token string) error {
	response := types.StreamResponse{
		ID:      s.requestID,
		Object:  "text_completion.chunk",
		Created: created,
		Model:   s.modelName,
		Choices: []types.StreamChoice{
			{
				Index: index,
				Text:  token,
			},
		},
	}

	return s.writeSSEData(response)
}

// writeCompletion writes the final completion token
func (s *SSEWriter) writeCompletion(created int64, index int, finishReason string) error {
	if finishReason == "" {
		finishReason = "stop"
	}

	response := types.StreamResponse{
		ID:      s.requestID,
		Object:  "text_completion.chunk",
		Created: created,
		Model:   s.modelName,
		Choices: []types.StreamChoice{
			{
				Index:        index,
				Text:         "",
				FinishReason: &finishReason,
			},
		},
	}

	return s.writeSSEData(response)
}

// writeError writes an error to the stream
func (s *SSEWriter) writeError(err error) error {
	errorResponse := types.ErrorResponse{
		Error: types.ErrorDetail{
			Message: err.Error(),
			Type:    "server_error",
			Code:    "500",
		},
	}

	return s.writeSSEData(errorResponse)
}

// writeDone writes the final "data: [DONE]" message
func (s *SSEWriter) writeDone() error {
	_, err := s.writer.Write([]byte("data: [DONE]\n\n"))
	if err != nil {
		return err
	}
	s.flusher.Flush()
	return nil
}

// writeSSEData writes data in SSE format
func (s *SSEWriter) writeSSEData(data interface{}) error {
	jsonData, err := json.Marshal(data)
	if err != nil {
		s.logger.WithError(err).Error("Failed to marshal JSON data")
		return fmt.Errorf("failed to marshal data: %w", err)
	}

	// Write in SSE format: "data: {json}\n\n"
	if _, err := s.writer.Write([]byte("data: ")); err != nil {
		return err
	}
	if _, err := s.writer.Write(jsonData); err != nil {
		return err
	}
	if _, err := s.writer.Write([]byte("\n\n")); err != nil {
		return err
	}

	s.flusher.Flush()
	return nil
}

// StreamingHandler is a helper function to handle streaming requests
type StreamingHandler struct {
	pool   *llm.WorkerPool
	logger *logrus.Logger
}

// NewStreamingHandler creates a new streaming handler
func NewStreamingHandler(pool *llm.WorkerPool, logger *logrus.Logger) *StreamingHandler {
	return &StreamingHandler{
		pool:   pool,
		logger: logger,
	}
}

// HandleStreamRequest handles a streaming completion request
func (h *StreamingHandler) HandleStreamRequest(w http.ResponseWriter, r *http.Request, req *types.CompletionRequest) {
	requestID := generateRequestID()
	modelInfo := h.pool.GetEngineStats()
	modelName := "llama" // Default model name

	// Create SSE writer
	sseWriter, err := NewSSEWriter(w, requestID, modelName, h.logger)
	if err != nil {
		http.Error(w, fmt.Sprintf("Streaming not supported: %v", err), http.StatusInternalServerError)
		return
	}

	// Submit streaming request to worker pool
	tokenChan, err := h.pool.SubmitStream(r.Context(), req)
	if err != nil {
		h.logger.WithError(err).Error("Failed to submit streaming request")
		sseWriter.writeError(fmt.Errorf("failed to process request: %w", err))
		return
	}

	// Stream the response
	if err := sseWriter.StreamCompletion(r.Context(), tokenChan); err != nil {
		if err != context.Canceled {
			h.logger.WithError(err).Error("Streaming error")
		}
		return
	}
}

// generateRequestID generates a unique request ID
func generateRequestID() string {
	return fmt.Sprintf("cmpl-%d", time.Now().UnixNano())
}

// IsStreamingRequest checks if the request is asking for streaming
func IsStreamingRequest(req *types.CompletionRequest) bool {
	return req != nil && req.Stream
}

// SetupStreamingHeaders sets up the necessary headers for streaming
func SetupStreamingHeaders(w http.ResponseWriter) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Headers", "Cache-Control")
	w.Header().Set("X-Accel-Buffering", "no") // Disable nginx buffering
}

// WriteSSEMessage writes a single SSE message
func WriteSSEMessage(w io.Writer, eventType, data string) error {
	if eventType != "" {
		if _, err := fmt.Fprintf(w, "event: %s\n", eventType); err != nil {
			return err
		}
	}
	if _, err := fmt.Fprintf(w, "data: %s\n\n", data); err != nil {
		return err
	}

	if flusher, ok := w.(http.Flusher); ok {
		flusher.Flush()
	}

	return nil
}

// WriteSSEJSON writes JSON data as SSE message
func WriteSSEJSON(w io.Writer, data interface{}) error {
	jsonData, err := json.Marshal(data)
	if err != nil {
		return fmt.Errorf("failed to marshal JSON: %w", err)
	}

	return WriteSSEMessage(w, "", string(jsonData))
}

// WriteSSEError writes an error as SSE message
func WriteSSEError(w io.Writer, err error, errorType, code string) error {
	errorResponse := types.ErrorResponse{
		Error: types.ErrorDetail{
			Message: err.Error(),
			Type:    errorType,
			Code:    code,
		},
	}

	return WriteSSEJSON(w, errorResponse)
}

// WriteSSEDone writes the final [DONE] message
func WriteSSEDone(w io.Writer) error {
	return WriteSSEMessage(w, "", "[DONE]")
}
