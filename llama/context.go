package llama

import "C"
import (
	"fmt"
	"runtime"
)

func NewContextWithModel(model *Model, params ContextParams) (*Context, error) {
	if model == nil || model.ptr == nil {
		return nil, fmt.Errorf("model is nil")
	}

	cParams := C.struct_llama_context_params{
		n_ctx:             C.uint32_t(params.NCtx),
		n_batch:           C.uint32_t(params.NBatch),
		n_ubatch:          C.uint32_t(params.NUBatch),
		n_seq_max:         C.uint32_t(params.NSeqMax),
		n_threads:         C.int32_t(params.NThreads),
		n_threads_batch:   C.int32_t(params.NThreadsBatch),
		rope_scaling_type: C.enum_llama_rope_scaling_type(params.RopeScalingType),
		pooling_type:      C.enum_llama_pooling_type(params.PoolingType),
		attention_type:    C.enum_llama_attention_type(params.AttentionType),
		rope_freq_base:    C.float(params.RopeFreqBase),
		rope_freq_scale:   C.float(params.RopeFreqScale),
		flash_attn:        C.bool(params.FlashAttn),
		offload_kqv:       C.bool(params.OffloadKQV),
		embeddings:        C.bool(params.Embeddings),
		no_perf:           C.bool(params.NoPerf),
		type_k:            C.enum_ggml_type(params.TypeK),
		type_v:            C.enum_ggml_type(params.TypeV),
	}

	cCtx := C.llama_init_from_model(model.ptr, cParams)
	if cCtx == nil {
		return nil, fmt.Errorf("failed to create context")
	}

	ctx := &Context{ptr: cCtx}
	runtime.SetFinalizer(ctx, (*Context).Free)
	return ctx, nil
}

func (ctx *Context) Free() {
	if ctx.ptr != nil {
		C.llama_free(ctx.ptr)
		ctx.ptr = nil
	}
}

// LLama aggregate helpers
func (llm *LLama) Free() {
	if llm == nil {
		return
	}
	if llm.Context != nil {
		llm.Context.Free()
	}
	if llm.Model != nil {
		llm.Model.Free()
	}
}

func (ctx *Context) Memory() *Memory {
	cMem := C.llama_get_memory(ctx.ptr)
	return &Memory{ptr: cMem}
}
