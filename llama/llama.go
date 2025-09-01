package llama

/*
#cgo CFLAGS: -I${SRCDIR}/../include
#cgo CXXFLAGS: -I${SRCDIR}/../include -std=c++17
#cgo LDFLAGS: -L${SRCDIR}/../lib -lllama -lggml -lm -pthread
#cgo darwin LDFLAGS: -framework Accelerate
#cgo linux LDFLAGS: -lrt

#include "llama.h"
*/
import "C"

// Backend init/free
func BackendInit() { C.llama_backend_init() }
func BackendFree() { C.llama_backend_free() }
