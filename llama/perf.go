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

func (ctx *Context) PerfContext() PerfContextData {
	c := C.llama_perf_context(ctx.ptr)
	return PerfContextData{
		TStartMs: float64(c.t_start_ms),
		TLoadMs:  float64(c.t_load_ms),
		TPEvalMs: float64(c.t_p_eval_ms),
		TEvalMs:  float64(c.t_eval_ms),
		NPEval:   int32(c.n_p_eval),
		NEval:    int32(c.n_eval),
		NReused:  int32(c.n_reused),
	}
}

func (ctx *Context) PerfReset() { C.llama_perf_context_reset(ctx.ptr) }
