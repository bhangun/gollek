package llama

/*
#cgo CFLAGS: -I${SRCDIR}/../include -I${SRCDIR}/../include/ggml
#cgo CXXFLAGS: -I${SRCDIR}/../include -I${SRCDIR}/../include/ggml -std=c++17
#cgo LDFLAGS: -L${SRCDIR}/../lib -lllama -lggml -lm -pthread
#cgo darwin LDFLAGS: -framework Accelerate
#cgo linux LDFLAGS: -lrt

#include "llama.h"
*/
import "C"
