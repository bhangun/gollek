package mcp

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/bhangun/wayang-inference/internal/llm"
	"github.com/bhangun/wayang-inference/pkg/types"

	"github.com/gin-gonic/gin"
	"github.com/gorilla/websocket"
	"github.com/sirupsen/logrus"
)

// MCPServer implements Model Context Protocol server for Go LLM inference server
type MCPServer struct {
	pool     *llm.WorkerPool
	logger   *logrus.Entry
	upgrader websocket.Upgrader
	clients  map[string]*MCPClient
}

// MCPClient represents a connected MCP client
type MCPClient struct {
	ID       string
	Conn     *websocket.Conn
	Send     chan []byte
	Server   *MCPServer
	LastPing time.Time
}

// MCPMessage represents a generic MCP protocol message
type MCPMessage struct {
	JSONRPC string      `json:"jsonrpc"`
	ID      interface{} `json:"id,omitempty"`
	Method  string      `json:"method,omitempty"`
	Params  interface{} `json:"params,omitempty"`
	Result  interface{} `json:"result,omitempty"`
	Error   *MCPError   `json:"error,omitempty"`
}

// MCPError represents an MCP error
type MCPError struct {
	Code    int         `json:"code"`
	Message string      `json:"message"`
	Data    interface{} `json:"data,omitempty"`
}

// MCPInferenceRequest represents an MCP inference request
type MCPInferenceRequest struct {
	Model       string                 `json:"model"`
	Messages    []MCPMessage           `json:"messages,omitempty"`
	Prompt      string                 `json:"prompt,omitempty"`
	MaxTokens   int                    `json:"max_tokens,omitempty"`
	Temperature float32                `json:"temperature,omitempty"`
	Stream      bool                   `json:"stream,omitempty"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
}

// MCPInferenceResponse represents an MCP inference response
type MCPInferenceResponse struct {
	Model     string                 `json:"model"`
	Response  string                 `json:"response"`
	Usage     types.Usage            `json:"usage"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
	Streaming bool                   `json:"streaming,omitempty"`
}

// NewMCPServer creates a new MCP server instance
func NewMCPServer(pool *llm.WorkerPool, logger *logrus.Logger) *MCPServer {
	return &MCPServer{
		pool:   pool,
		logger: logger.WithField("component", "mcp-server"),
		upgrader: websocket.Upgrader{
			CheckOrigin: func(r *http.Request) bool {
				return true // Allow all origins for MCP
			},
		},
		clients: make(map[string]*MCPClient),
	}
}

// SetupMCPRoutes sets up MCP-specific routes
func (s *MCPServer) SetupMCPRoutes(router *gin.Engine) {
	mcpGroup := router.Group("/mcp")
	{
		// WebSocket endpoint for MCP clients
		mcpGroup.GET("/ws", s.handleWebSocket)

		// HTTP endpoints for compatibility
		mcpGroup.POST("/inference", s.handleHTTPInference)
		mcpGroup.GET("/capabilities", s.handleCapabilities)
		mcpGroup.GET("/models", s.handleModels)

		// Server information
		mcpGroup.GET("/info", s.handleServerInfo)
	}
}

// handleWebSocket handles WebSocket connections from MCP clients
func (s *MCPServer) handleWebSocket(c *gin.Context) {
	conn, err := s.upgrader.Upgrade(c.Writer, c.Request, nil)
	if err != nil {
		s.logger.WithError(err).Error("Failed to upgrade WebSocket connection")
		return
	}

	clientID := generateClientID()
	client := &MCPClient{
		ID:       clientID,
		Conn:     conn,
		Send:     make(chan []byte, 256),
		Server:   s,
		LastPing: time.Now(),
	}

	s.clients[clientID] = client
	s.logger.WithField("client_id", clientID).Info("MCP client connected")

	// Start goroutines for handling client
	go client.readPump()
	go client.writePump()
}

// handleHTTPInference handles HTTP-based inference requests
func (s *MCPServer) handleHTTPInference(c *gin.Context) {
	var mcpReq MCPInferenceRequest
	if err := c.ShouldBindJSON(&mcpReq); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request format"})
		return
	}

	// Convert MCP request to internal format
	req := s.convertMCPToInternal(&mcpReq)

	if mcpReq.Stream {
		s.handleStreamingInference(c, req, &mcpReq)
	} else {
		s.handleSyncInference(c, req, &mcpReq)
	}
}

// handleCapabilities returns server capabilities
func (s *MCPServer) handleCapabilities(c *gin.Context) {
	capabilities := gin.H{
		"protocol_version": "2024-11-05",
		"server_info": gin.H{
			"name":    "github.com/bhangun/wayang-inference",
			"version": "1.0.0",
		},
		"capabilities": gin.H{
			"inference": gin.H{
				"supports_streaming": true,
				"supports_tools":     false,
				"max_tokens":         4096,
			},
			"resources": gin.H{
				"supports_templates": false,
			},
			"tools": gin.H{
				"supports_calls": false,
			},
		},
	}
	c.JSON(http.StatusOK, capabilities)
}

// handleModels returns available models
func (s *MCPServer) handleModels(c *gin.Context) {
	engineStats := s.pool.GetEngineStats()
	models := []gin.H{
		{
			"id":      "llama-local",
			"name":    "Local Llama Model",
			"loaded":  engineStats.ModelLoaded,
			"type":    "text-generation",
			"context": 2048, // Would get from config
		},
	}
	c.JSON(http.StatusOK, gin.H{"models": models})
}

// handleServerInfo returns server information
func (s *MCPServer) handleServerInfo(c *gin.Context) {
	poolStats := s.pool.GetStats()
	engineStats := s.pool.GetEngineStats()

	info := gin.H{
		"server": gin.H{
			"name":    "github.com/bhangun/wayang-inference",
			"version": "1.0.0",
			"status":  "running",
		},
		"mcp": gin.H{
			"protocol_version":  "2024-11-05",
			"connected_clients": len(s.clients),
		},
		"inference": gin.H{
			"model_loaded":    engineStats.ModelLoaded,
			"active_requests": poolStats.ActiveRequests,
			"queued_requests": poolStats.QueuedRequests,
			"total_requests":  poolStats.TotalRequests,
		},
	}
	c.JSON(http.StatusOK, info)
}

// handleSyncInference handles synchronous inference requests
func (s *MCPServer) handleSyncInference(c *gin.Context, req *types.CompletionRequest, mcpReq *MCPInferenceRequest) {
	response, err := s.pool.Submit(c.Request.Context(), req)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	mcpResponse := s.convertInternalToMCP(response, mcpReq)
	c.JSON(http.StatusOK, mcpResponse)
}

// handleStreamingInference handles streaming inference requests
func (s *MCPServer) handleStreamingInference(c *gin.Context, req *types.CompletionRequest, mcpReq *MCPInferenceRequest) {
	// Set SSE headers
	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")
	c.Header("Connection", "keep-alive")

	tokenChan, err := s.pool.SubmitStream(c.Request.Context(), req)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	// Stream tokens
	for token := range tokenChan {
		if token.Error != nil {
			break
		}

		mcpChunk := gin.H{
			"model":     mcpReq.Model,
			"chunk":     token.Token,
			"streaming": true,
			"done":      token.IsComplete,
		}

		data, _ := json.Marshal(mcpChunk)
		c.SSEvent("data", string(data))
		c.Writer.Flush()

		if token.IsComplete {
			break
		}
	}
}

// Client methods

// readPump handles incoming messages from MCP clients
func (c *MCPClient) readPump() {
	defer func() {
		c.Server.unregisterClient(c)
		c.Conn.Close()
	}()

	c.Conn.SetReadLimit(512 * 1024) // 512KB max message size
	c.Conn.SetReadDeadline(time.Now().Add(60 * time.Second))
	c.Conn.SetPongHandler(func(string) error {
		c.Conn.SetReadDeadline(time.Now().Add(60 * time.Second))
		c.LastPing = time.Now()
		return nil
	})

	for {
		_, message, err := c.Conn.ReadMessage()
		if err != nil {
			c.Server.logger.WithError(err).WithField("client_id", c.ID).Error("WebSocket read error")
			break
		}

		c.handleMessage(message)
	}
}

// writePump handles outgoing messages to MCP clients
func (c *MCPClient) writePump() {
	ticker := time.NewTicker(54 * time.Second)
	defer func() {
		ticker.Stop()
		c.Conn.Close()
	}()

	for {
		select {
		case message, ok := <-c.Send:
			c.Conn.SetWriteDeadline(time.Now().Add(10 * time.Second))
			if !ok {
				c.Conn.WriteMessage(websocket.CloseMessage, []byte{})
				return
			}

			if err := c.Conn.WriteMessage(websocket.TextMessage, message); err != nil {
				c.Server.logger.WithError(err).WithField("client_id", c.ID).Error("WebSocket write error")
				return
			}

		case <-ticker.C:
			c.Conn.SetWriteDeadline(time.Now().Add(10 * time.Second))
			if err := c.Conn.WriteMessage(websocket.PingMessage, nil); err != nil {
				return
			}
		}
	}
}

// handleMessage processes incoming MCP messages
func (c *MCPClient) handleMessage(message []byte) {
	var mcpMsg MCPMessage
	if err := json.Unmarshal(message, &mcpMsg); err != nil {
		c.sendError(-32700, "Parse error", nil, mcpMsg.ID)
		return
	}

	switch mcpMsg.Method {
	case "initialize":
		c.handleInitialize(&mcpMsg)
	case "inference/generate":
		c.handleInferenceGenerate(&mcpMsg)
	case "ping":
		c.handlePing(&mcpMsg)
	default:
		c.sendError(-32601, "Method not found", nil, mcpMsg.ID)
	}
}

// handleInitialize handles client initialization
func (c *MCPClient) handleInitialize(msg *MCPMessage) {
	response := MCPMessage{
		JSONRPC: "2.0",
		ID:      msg.ID,
		Result: gin.H{
			"protocol_version": "2024-11-05",
			"server_info": gin.H{
				"name":    "github.com/bhangun/wayang-inference",
				"version": "1.0.0",
			},
			"capabilities": gin.H{
				"inference": gin.H{
					"supports_streaming": true,
				},
			},
		},
	}

	c.sendMessage(&response)
	c.Server.logger.WithField("client_id", c.ID).Info("Client initialized")
}

// handleInferenceGenerate handles inference requests
func (c *MCPClient) handleInferenceGenerate(msg *MCPMessage) {
	// Parse inference request
	paramsBytes, err := json.Marshal(msg.Params)
	if err != nil {
		c.sendError(-32602, "Invalid params", nil, msg.ID)
		return
	}

	var mcpReq MCPInferenceRequest
	if err := json.Unmarshal(paramsBytes, &mcpReq); err != nil {
		c.sendError(-32602, "Invalid inference request", nil, msg.ID)
		return
	}

	// Convert to internal format
	req := c.Server.convertMCPToInternal(&mcpReq)

	// Submit to worker pool
	if mcpReq.Stream {
		c.handleStreamingRequest(msg.ID, req, &mcpReq)
	} else {
		c.handleSyncRequest(msg.ID, req, &mcpReq)
	}
}

// Helper methods

func (s *MCPServer) convertMCPToInternal(mcpReq *MCPInferenceRequest) *types.CompletionRequest {
	prompt := mcpReq.Prompt
	if prompt == "" && len(mcpReq.Messages) > 0 {
		// Convert messages to prompt (simplified)
		for _, msg := range mcpReq.Messages {
			if msgStr, ok := msg.Params.(string); ok {
				prompt += msgStr + "\n"
			}
		}
	}

	return &types.CompletionRequest{
		Prompt:      prompt,
		MaxTokens:   mcpReq.MaxTokens,
		Temperature: mcpReq.Temperature,
		Stream:      mcpReq.Stream,
	}
}

func (s *MCPServer) convertInternalToMCP(response *types.CompletionResponse, mcpReq *MCPInferenceRequest) *MCPInferenceResponse {
	text := ""
	if len(response.Choices) > 0 {
		text = response.Choices[0].Text
	}

	return &MCPInferenceResponse{
		Model:    mcpReq.Model,
		Response: text,
		Usage:    response.Usage,
		Metadata: mcpReq.Metadata,
	}
}

func (c *MCPClient) sendMessage(msg *MCPMessage) {
	data, err := json.Marshal(msg)
	if err != nil {
		c.Server.logger.WithError(err).Error("Failed to marshal message")
		return
	}

	select {
	case c.Send <- data:
	default:
		close(c.Send)
	}
}

func (c *MCPClient) sendError(code int, message string, data interface{}, id interface{}) {
	errorMsg := MCPMessage{
		JSONRPC: "2.0",
		ID:      id,
		Error: &MCPError{
			Code:    code,
			Message: message,
			Data:    data,
		},
	}
	c.sendMessage(&errorMsg)
}

func (s *MCPServer) unregisterClient(client *MCPClient) {
	delete(s.clients, client.ID)
	s.logger.WithField("client_id", client.ID).Info("MCP client disconnected")
}

func generateClientID() string {
	return fmt.Sprintf("client-%d", time.Now().UnixNano())
}

// Additional method implementations for streaming and sync requests
func (c *MCPClient) handleSyncRequest(msgID interface{}, req *types.CompletionRequest, mcpReq *MCPInferenceRequest) {
	ctx := context.Background()
	response, err := c.Server.pool.Submit(ctx, req)
	if err != nil {
		c.sendError(-32603, "Internal error", err.Error(), msgID)
		return
	}

	mcpResponse := c.Server.convertInternalToMCP(response, mcpReq)
	result := MCPMessage{
		JSONRPC: "2.0",
		ID:      msgID,
		Result:  mcpResponse,
	}
	c.sendMessage(&result)
}

func (c *MCPClient) handleStreamingRequest(msgID interface{}, req *types.CompletionRequest, mcpReq *MCPInferenceRequest) {
	ctx := context.Background()
	tokenChan, err := c.Server.pool.SubmitStream(ctx, req)
	if err != nil {
		c.sendError(-32603, "Internal error", err.Error(), msgID)
		return
	}

	// Start streaming in a separate goroutine
	go func() {
		for token := range tokenChan {
			if token.Error != nil {
				c.sendError(-32603, "Stream error", token.Error.Error(), msgID)
				return
			}

			streamMsg := MCPMessage{
				JSONRPC: "2.0",
				ID:      msgID,
				Result: gin.H{
					"chunk":     token.Token,
					"streaming": true,
					"done":      token.IsComplete,
				},
			}
			c.sendMessage(&streamMsg)

			if token.IsComplete {
				break
			}
		}
	}()
}

func (c *MCPClient) handlePing(msg *MCPMessage) {
	pong := MCPMessage{
		JSONRPC: "2.0",
		ID:      msg.ID,
		Result:  "pong",
	}
	c.sendMessage(&pong)
}
