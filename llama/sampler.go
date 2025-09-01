package llama

/*
#include "llama.h"
*/
import "C"
import "runtime"

func NewSamplerChain(params C.struct_llama_sampler_chain_params) *Sampler {
	c := C.llama_sampler_chain_init(params)
	s := &Sampler{ptr: c}
	runtime.SetFinalizer(s, (*Sampler).Free)
	return s
}

func NewSamplerGreedy() *Sampler { return wrapSampler(C.llama_sampler_init_greedy()) }
func NewSamplerDist(seed uint32) *Sampler {
	return wrapSampler(C.llama_sampler_init_dist(C.uint32_t(seed)))
}
func NewSamplerTopK(k int32) *Sampler { return wrapSampler(C.llama_sampler_init_top_k(C.int32_t(k))) }
func NewSamplerTopP(p float32, minKeep int) *Sampler {
	return wrapSampler(C.llama_sampler_init_top_p(C.float(p), C.size_t(minKeep)))
}
func NewSamplerMinP(p float32, minKeep int) *Sampler {
	return wrapSampler(C.llama_sampler_init_min_p(C.float(p), C.size_t(minKeep)))
}
func NewSamplerTemp(temperature float32) *Sampler {
	return wrapSampler(C.llama_sampler_init_temp(C.float(temperature)))
}
func NewSamplerTempExt(temp, delta, exponent float32) *Sampler {
	return wrapSampler(C.llama_sampler_init_temp_ext(C.float(temp), C.float(delta), C.float(exponent)))
}
func NewSamplerMirostat(nVocab int32, seed uint32, tau, eta float32, m int32) *Sampler {
	return wrapSampler(C.llama_sampler_init_mirostat(C.int32_t(nVocab), C.uint32_t(seed), C.float(tau), C.float(eta), C.int32_t(m)))
}
func NewSamplerMirostatV2(seed uint32, tau, eta float32) *Sampler {
	return wrapSampler(C.llama_sampler_init_mirostat_v2(C.uint32_t(seed), C.float(tau), C.float(eta)))
}
func NewSamplerPenalties(lastN int32, repeat, freq, present float32) *Sampler {
	return wrapSampler(C.llama_sampler_init_penalties(C.int32_t(lastN), C.float(repeat), C.float(freq), C.float(present)))
}
func NewSamplerTypical(p float32, minKeep int) *Sampler {
	return wrapSampler(C.llama_sampler_init_typical(C.float(p), C.size_t(minKeep)))
}

func wrapSampler(c *C.struct_llama_sampler) *Sampler {
	s := &Sampler{ptr: c}
	runtime.SetFinalizer(s, (*Sampler).Free)
	return s
}

func (chain *Sampler) ChainAdd(s *Sampler) {
	if chain == nil || chain.ptr == nil || s == nil || s.ptr == nil {
		return
	}
	C.llama_sampler_chain_add(chain.ptr, s.ptr)
	runtime.SetFinalizer(s, nil)
	s.ptr = nil
}

func (s *Sampler) Sample(ctx *Context, idx int32) Token {
	return Token(C.llama_sampler_sample(s.ptr, ctx.ptr, C.int32_t(idx)))
}
func (s *Sampler) Accept(t Token)  { C.llama_sampler_accept(s.ptr, C.llama_token(t)) }
func (s *Sampler) Reset()          { C.llama_sampler_reset(s.ptr) }
func (s *Sampler) GetSeed() uint32 { return uint32(C.llama_sampler_get_seed(s.ptr)) }
func (s *Sampler) Free() {
	if s.ptr != nil {
		C.llama_sampler_free(s.ptr)
		s.ptr = nil
	}
}
