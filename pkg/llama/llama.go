package llama

/*
#cgo CFLAGS: -I${SRCDIR}/../../include
#cgo LDFLAGS: -L${SRCDIR}/../../lib -lllama
#include "llama.h"
#include <stdlib.h>
*/
import "C"
import (
	"fmt"
	"unsafe"
)

type LLama struct {
	ctx *C.struct_llama_context
}

// ModelOption is just a config function
type ModelOption func(*C.struct_llama_context_params)

// PredictOption configures prediction
type PredictOption func(*C.struct_llama_sampling_params)

func New(modelPath string, opts ...ModelOption) (*LLama, error) {
	cPath := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cPath))

	params := C.llama_context_default_params()
	for _, opt := range opts {
		opt(&params)
	}

	ctx := C.llama_init_from_file(cPath, params)
	if ctx == nil {
		return nil, fmt.Errorf("failed to load model: %s", modelPath)
	}
	return &LLama{ctx: ctx}, nil
}

func (l *LLama) Predict(prompt string, opts ...PredictOption) (string, error) {
	// 🚧 simplified placeholder
	// You’d need to allocate tokens, run llama_eval, decode them, etc.
	return "[unimplemented prediction]", nil
}

func (l *LLama) Free() {
	if l.ctx != nil {
		C.llama_free(l.ctx)
		l.ctx = nil
	}
}

// Example option: set context size
func SetContext(n int) ModelOption {
	return func(p *C.struct_llama_context_params) {
		p.n_ctx = C.int(n)
	}
}
