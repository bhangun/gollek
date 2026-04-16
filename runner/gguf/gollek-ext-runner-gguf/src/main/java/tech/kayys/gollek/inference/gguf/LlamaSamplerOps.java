package tech.kayys.gollek.inference.gguf;

import java.lang.foreign.*;

/**
 * Wraps all llama.cpp sampler-chain operations.
 *
 * <p>A sampler chain is built by calling {@link #createChain()} and then adding
 * one or more sampler stages. The chain is passed to {@link #sample} to draw the
 * next token, and freed with {@link #freeChain} when done.
 *
 * <p>Optional samplers (top-p, min-p, Mirostat, grammar, typical) are only
 * available when the native library exports the corresponding symbol; calling
 * them on an unsupported build throws {@link IllegalStateException}.
 */
final class LlamaSamplerOps {

    private final LlamaHandles h;
    private final Arena arena;

    LlamaSamplerOps(LlamaHandles handles, Arena arena) {
        this.h = handles;
        this.arena = arena;
    }

    /**
     * Creates a new sampler chain with default parameters.
     *
     * @return a native sampler chain handle
     */
    MemorySegment createChain() {
        try {
            MemorySegment params = arena.allocate(LlamaStructLayouts.SAMPLER_CHAIN_PARAMS);
            params.set(ValueLayout.JAVA_BYTE, 0, (byte) 0); // no_perf = false
            return (MemorySegment) h.samplerChainInit.invoke(params);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to create sampler chain", e);
        }
    }

    /** Adds a greedy (argmax) sampler to the chain. */
    void addGreedy(MemorySegment chain) {
        try {
            h.samplerChainAdd.invoke(chain, (MemorySegment) h.samplerInitGreedy.invoke());
        } catch (Throwable e) {
            throw new RuntimeException("Failed to add greedy sampler", e);
        }
    }

    /**
     * Adds a top-k sampler to the chain.
     *
     * @param k number of top candidates to keep
     */
    void addTopK(MemorySegment chain, int k) {
        try {
            h.samplerChainAdd.invoke(chain, (MemorySegment) h.samplerInitTopK.invoke(k));
        } catch (Throwable e) {
            throw new RuntimeException("Failed to add top-k sampler", e);
        }
    }

    /**
     * Adds a nucleus (top-p) sampler to the chain.
     *
     * @param p       cumulative probability threshold in {@code (0, 1]}
     * @param minKeep minimum number of candidates to retain
     */
    void addTopP(MemorySegment chain, float p, long minKeep) {
        try {
            h.require(h.samplerInitTopP, "llama_sampler_init_top_p");
            h.samplerChainAdd.invoke(chain, (MemorySegment) h.samplerInitTopP.invoke(p, minKeep));
        } catch (Throwable e) {
            throw new RuntimeException("Failed to add top-p sampler", e);
        }
    }

    /**
     * Adds a min-p sampler to the chain.
     *
     * @param p       minimum probability relative to the top token
     * @param minKeep minimum number of candidates to retain
     */
    void addMinP(MemorySegment chain, float p, long minKeep) {
        try {
            h.require(h.samplerInitMinP, "llama_sampler_init_min_p");
            h.samplerChainAdd.invoke(chain, (MemorySegment) h.samplerInitMinP.invoke(p, minKeep));
        } catch (Throwable e) {
            throw new RuntimeException("Failed to add min-p sampler", e);
        }
    }

    /**
     * Adds a temperature sampler to the chain.
     *
     * @param temp sampling temperature; lower = more deterministic
     */
    void addTemp(MemorySegment chain, float temp) {
        try {
            h.samplerChainAdd.invoke(chain, (MemorySegment) h.samplerInitTemp.invoke(temp));
        } catch (Throwable e) {
            throw new RuntimeException("Failed to add temperature sampler", e);
        }
    }

    /**
     * Adds a distribution sampler (random draw from the remaining candidates).
     *
     * @param seed RNG seed
     */
    void addDist(MemorySegment chain, int seed) {
        try {
            h.samplerChainAdd.invoke(chain, (MemorySegment) h.samplerInitDist.invoke(seed));
        } catch (Throwable e) {
            throw new RuntimeException("Failed to add dist sampler", e);
        }
    }

    /**
     * Adds a repetition/frequency/presence penalty sampler to the chain.
     *
     * @param penaltyLastN     number of last tokens to penalise (0 = disable, -1 = context size)
     * @param repeatPenalty    repetition penalty; 1.0 = disabled
     * @param frequencyPenalty frequency penalty; 0.0 = disabled
     * @param presencePenalty  presence penalty; 0.0 = disabled
     */
    void addPenalties(MemorySegment chain, int penaltyLastN,
            float repeatPenalty, float frequencyPenalty, float presencePenalty) {
        try {
            h.require(h.samplerInitPenalties, "llama_sampler_init_penalties");
            h.samplerChainAdd.invoke(chain,
                    (MemorySegment) h.samplerInitPenalties.invoke(
                            penaltyLastN, repeatPenalty, frequencyPenalty, presencePenalty));
        } catch (Throwable e) {
            throw new RuntimeException("Failed to add penalties sampler", e);
        }
    }

    /**
     * Adds a Mirostat v1 sampler to the chain.
     *
     * @param nVocab vocabulary size
     * @param seed   RNG seed
     * @param tau    target cross-entropy (perplexity target)
     * @param eta    learning rate
     * @param m      number of tokens for ŝ estimation
     */
    void addMirostat(MemorySegment chain, int nVocab, int seed, float tau, float eta, int m) {
        try {
            h.require(h.samplerInitMirostat, "llama_sampler_init_mirostat");
            h.samplerChainAdd.invoke(chain,
                    (MemorySegment) h.samplerInitMirostat.invoke(nVocab, seed, tau, eta, m));
        } catch (Throwable e) {
            throw new RuntimeException("Failed to add Mirostat sampler", e);
        }
    }

    /**
     * Adds a Mirostat v2 sampler to the chain.
     *
     * @param seed RNG seed
     * @param tau  target cross-entropy
     * @param eta  learning rate
     */
    void addMirostatV2(MemorySegment chain, int seed, float tau, float eta) {
        try {
            h.require(h.samplerInitMirostatV2, "llama_sampler_init_mirostat_v2");
            h.samplerChainAdd.invoke(chain,
                    (MemorySegment) h.samplerInitMirostatV2.invoke(seed, tau, eta));
        } catch (Throwable e) {
            throw new RuntimeException("Failed to add Mirostat v2 sampler", e);
        }
    }

    /**
     * Adds a GBNF grammar sampler to constrain output to a formal grammar.
     *
     * @param vocab       the model's vocab handle (from {@code llama_model_get_vocab})
     * @param grammarStr  GBNF grammar string
     * @param grammarRoot root rule name (typically {@code "root"})
     */
    void addGrammar(MemorySegment chain, MemorySegment vocab, String grammarStr, String grammarRoot) {
        try {
            h.require(h.samplerInitGrammar, "llama_sampler_init_grammar");
            MemorySegment gs = arena.allocateFrom(grammarStr);
            MemorySegment gr = arena.allocateFrom(grammarRoot);
            MemorySegment sampler = (MemorySegment) h.samplerInitGrammar.invoke(vocab, gs, gr);
            if (sampler == null || sampler.address() == 0) {
                throw new RuntimeException("Grammar parse failed for: "
                        + grammarStr.substring(0, Math.min(80, grammarStr.length())));
            }
            h.samplerChainAdd.invoke(chain, sampler);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to add grammar sampler", e);
        }
    }

    /**
     * Adds a locally-typical sampler to the chain.
     *
     * @param p       typical-p value
     * @param minKeep minimum candidates to retain
     */
    void addTypical(MemorySegment chain, float p, long minKeep) {
        try {
            h.require(h.samplerInitTypical, "llama_sampler_init_typical");
            h.samplerChainAdd.invoke(chain, (MemorySegment) h.samplerInitTypical.invoke(p, minKeep));
        } catch (Throwable e) {
            throw new RuntimeException("Failed to add typical sampler", e);
        }
    }

    /**
     * Samples the next token from the logits at the given batch index.
     *
     * @param chain   the sampler chain
     * @param context the inference context
     * @param index   batch position whose logits to sample from
     * @return the sampled token ID
     */
    int sample(MemorySegment chain, MemorySegment context, int index) {
        try {
            h.require(h.samplerSample, "llama_sampler_sample");
            return (int) h.samplerSample.invoke(chain, context, index);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to sample token", e);
        }
    }

    /**
     * Frees a sampler chain and all its constituent samplers.
     *
     * @param chain the sampler chain to free
     */
    void freeChain(MemorySegment chain) {
        try {
            h.samplerFree.invoke(chain);
        } catch (Throwable e) {
            // Log but don't rethrow — called from finally blocks
        }
    }
}
