package llama

/*
#include "llama.h"
*/
import "C"
import (
	"fmt"
	"runtime"
	"unsafe"
)

func LoadModelFromFile(path string, params ModelParams) (*Model, error) {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	cParams := C.struct_llama_model_params{
		n_gpu_layers:  C.int32_t(params.NGPULayers),
		split_mode:    C.enum_llama_split_mode(params.SplitMode),
		main_gpu:      C.int32_t(params.MainGPU),
		vocab_only:    C.bool(params.VocabOnly),
		use_mmap:      C.bool(params.UseMMmap),
		use_mlock:     C.bool(params.UseMLock),
		check_tensors: C.bool(params.CheckTensors),
	}

	cModel := C.llama_model_load_from_file(cPath, cParams)
	if cModel == nil {
		return nil, fmt.Errorf("failed to load model from file: %s", path)
	}

	model := &Model{ptr: cModel}
	runtime.SetFinalizer(model, (*Model).Free)
	return model, nil
}

func (model *Model) Free() {
	if model.ptr != nil {
		C.llama_model_free(model.ptr)
		model.ptr = nil
	}
}

// Model info
func (model *Model) VocabType() int {
	return int(C.llama_vocab_type(C.llama_model_get_vocab(model.ptr)))
}
func (model *Model) RopeType() int     { return int(C.llama_model_rope_type(model.ptr)) }
func (model *Model) NCtxTrain() int32  { return int32(C.llama_model_n_ctx_train(model.ptr)) }
func (model *Model) NEmbd() int32      { return int32(C.llama_model_n_embd(model.ptr)) }
func (model *Model) NLayer() int32     { return int32(C.llama_model_n_layer(model.ptr)) }
func (model *Model) NHead() int32      { return int32(C.llama_model_n_head(model.ptr)) }
func (model *Model) NHeadKV() int32    { return int32(C.llama_model_n_head_kv(model.ptr)) }
func (model *Model) NParams() uint64   { return uint64(C.llama_model_n_params(model.ptr)) }
func (model *Model) Size() uint64      { return uint64(C.llama_model_size(model.ptr)) }
func (model *Model) HasEncoder() bool  { return bool(C.llama_model_has_encoder(model.ptr)) }
func (model *Model) HasDecoder() bool  { return bool(C.llama_model_has_decoder(model.ptr)) }
func (model *Model) IsRecurrent() bool { return bool(C.llama_model_is_recurrent(model.ptr)) }
func (model *Model) RopeFreqScaleTrain() float32 {
	return float32(C.llama_model_rope_freq_scale_train(model.ptr))
}

func (model *Model) ChatTemplate(name string) string {
	var cName *C.char
	if name != "" {
		cName = C.CString(name)
		defer C.free(unsafe.Pointer(cName))
	}
	cTemplate := C.llama_model_chat_template(model.ptr, cName)
	if cTemplate == nil {
		return ""
	}
	return C.GoString(cTemplate)
}

func (model *Model) Vocab() *Vocab {
	cV := C.llama_model_get_vocab(model.ptr)
	return &Vocab{ptr: cV}
}
