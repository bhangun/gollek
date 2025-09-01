github.com/bhangun/wayang-inference/
├── cmd/
│   └── server/
│       └── main.go              # Application entry point
├── internal/
│   ├── config/
│   │   └── config.go            # Configuration management
│   ├── server/
│   │   ├── server.go            # HTTP server setup
│   │   ├── handlers.go          # HTTP handlers
│   │   └── middleware.go        # Custom middleware
│   ├── llm/
│   │   ├── engine.go            # LLM engine interface
│   │   ├── llama.go             # llama.cpp implementation
│   │   └── pool.go              # Worker pool for concurrency
│   ├── streaming/
│   │   └── sse.go               # Server-Sent Events implementation
│   └── health/
│       └── health.go            # Health check implementation
├── pkg/
│   ├── types/
│   │   └── api.go               # API request/response types
│   └── logger/
│       └── logger.go            # Structured logging
├── build/
│   ├── Dockerfile              # Container build
│   ├── Makefile                # Build automation
│   └── gpu/
│       ├── cuda.mk             # CUDA build configuration
│       └── metal.mk            # Metal build configuration
├── configs/
│   ├── config.yaml             # Default configuration
│   └── config.example.yaml     # Configuration template
├── scripts/
│   ├── build-cpu.sh            # CPU-only build script
│   ├── build-gpu-cuda.sh       # CUDA build script
│   └── build-gpu-metal.sh      # Metal build script
├── docs/
│   ├── API.md                  # API documentation
│   └── DEPLOYMENT.md           # Deployment guide
├── go.mod
├── go.sum
└── README.md