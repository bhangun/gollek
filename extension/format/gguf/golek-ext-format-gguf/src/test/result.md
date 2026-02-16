olek-cli % java -jar target/golek-cli-1.0.0-SNAPSHOT-runner.jar run --provider gguf --model Qwen/Qwen2.5-0.5B-Instruct --prompt "Hello"
Checking model: Qwen/Qwen2.5-0.5B-Instruct... found locally.
Model path: file://~/.golek/models/gguf/Qwen_Qwen2.5-0.5B-Instruct-GGUF
Running inference [Model: Qwen/Qwen2.5-0.5B-Instruct, Tenant: default]
Provider: gguf
ggml_metal_device_init: tensor API disabled for pre-M5 and pre-A19 devices
ggml_metal_library_init: using embedded metal library
ggml_metal_library_init: loaded in 0.020 sec
ggml_metal_rsets_init: creating a residency set collection (keep_alive = 180 s)
ggml_metal_device_init: GPU name:   MTL0
ggml_metal_device_init: GPU family: MTLGPUFamilyApple9  (1009)
ggml_metal_device_init: GPU family: MTLGPUFamilyCommon3 (3003)
ggml_metal_device_init: GPU family: MTLGPUFamilyMetal4  (5002)
ggml_metal_device_init: simdgroup reduction   = true
ggml_metal_device_init: simdgroup matrix mul. = true
ggml_metal_device_init: has unified memory    = true
ggml_metal_device_init: has bfloat            = true
ggml_metal_device_init: has tensor            = false
ggml_metal_device_init: use residency sets    = true
ggml_metal_device_init: use shared buffers    = true
ggml_metal_device_init: recommendedMaxWorkingSetSize  = 12713.12 MB
llama_model_load_from_file_impl: using device MTL0 (Apple M4) (unknown id) - 12123 MiB free
llama_model_loader: loaded meta data with 26 key-value pairs and 291 tensors from ~/.golek/models/gguf/Qwen_Qwen2.5-0.5B-Instruct-GGUF (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = qwen2
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = qwen2.5-0.5b-instruct
llama_model_loader: - kv   3:                            general.version str              = v0.1
llama_model_loader: - kv   4:                           general.finetune str              = qwen2.5-0.5b-instruct
llama_model_loader: - kv   5:                         general.size_label str              = 630M
llama_model_loader: - kv   6:                          qwen2.block_count u32              = 24
llama_model_loader: - kv   7:                       qwen2.context_length u32              = 32768
llama_model_loader: - kv   8:                     qwen2.embedding_length u32              = 896
llama_model_loader: - kv   9:                  qwen2.feed_forward_length u32              = 4864
llama_model_loader: - kv  10:                 qwen2.attention.head_count u32              = 14
llama_model_loader: - kv  11:              qwen2.attention.head_count_kv u32              = 2
llama_model_loader: - kv  12:                       qwen2.rope.freq_base f32              = 1000000.000000
llama_model_loader: - kv  13:     qwen2.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  14:                          general.file_type u32              = 15
llama_model_loader: - kv  15:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  16:                         tokenizer.ggml.pre str              = qwen2
llama_model_loader: - kv  17:                      tokenizer.ggml.tokens arr[str,151936]  = ["!", "\"", "#", "$", "%", "&", "'", ...
llama_model_loader: - kv  18:                  tokenizer.ggml.token_type arr[i32,151936]  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  19:                      tokenizer.ggml.merges arr[str,151387]  = ["Ġ Ġ", "ĠĠ ĠĠ", "i n", "Ġ t",...
llama_model_loader: - kv  20:                tokenizer.ggml.eos_token_id u32              = 151645
llama_model_loader: - kv  21:            tokenizer.ggml.padding_token_id u32              = 151643
llama_model_loader: - kv  22:                tokenizer.ggml.bos_token_id u32              = 151643
llama_model_loader: - kv  23:               tokenizer.ggml.add_bos_token bool             = false
llama_model_loader: - kv  24:                    tokenizer.chat_template str              = {%- if tools %}\n    {{- '<|im_start|>...
llama_model_loader: - kv  25:               general.quantization_version u32              = 2
llama_model_loader: - type  f32:  121 tensors
llama_model_loader: - type q5_0:  133 tensors
llama_model_loader: - type q8_0:   13 tensors
llama_model_loader: - type q4_K:   12 tensors
llama_model_loader: - type q6_K:   12 tensors
print_info: file format = GGUF V3 (latest)
print_info: file type   = Q4_K - Medium
print_info: file size   = 462.96 MiB (6.16 BPW) 
init_tokenizer: initializing tokenizer for type 2
load: 0 unused tokens
load: control token: 151659 '<|fim_prefix|>' is not marked as EOG
load: control token: 151656 '<|video_pad|>' is not marked as EOG
load: control token: 151655 '<|image_pad|>' is not marked as EOG
load: control token: 151653 '<|vision_end|>' is not marked as EOG
load: control token: 151652 '<|vision_start|>' is not marked as EOG
load: control token: 151651 '<|quad_end|>' is not marked as EOG
load: control token: 151649 '<|box_end|>' is not marked as EOG
load: control token: 151648 '<|box_start|>' is not marked as EOG
load: control token: 151646 '<|object_ref_start|>' is not marked as EOG
load: control token: 151644 '<|im_start|>' is not marked as EOG
load: control token: 151661 '<|fim_suffix|>' is not marked as EOG
load: control token: 151647 '<|object_ref_end|>' is not marked as EOG
load: control token: 151660 '<|fim_middle|>' is not marked as EOG
load: control token: 151654 '<|vision_pad|>' is not marked as EOG
load: control token: 151650 '<|quad_start|>' is not marked as EOG
load: printing all EOG tokens:
load:   - 151643 ('<|endoftext|>')
load:   - 151645 ('<|im_end|>')
load:   - 151662 ('<|fim_pad|>')
load:   - 151663 ('<|repo_name|>')
load:   - 151664 ('<|file_sep|>')
load: special tokens cache size = 22
load: token to piece cache size = 0.9310 MB
print_info: arch                  = qwen2
print_info: vocab_only            = 0
print_info: no_alloc              = 0
print_info: n_ctx_train           = 32768
print_info: n_embd                = 896
print_info: n_embd_inp            = 896
print_info: n_layer               = 24
print_info: n_head                = 14
print_info: n_head_kv             = 2
print_info: n_rot                 = 64
print_info: n_swa                 = 0
print_info: is_swa_any            = 0
print_info: n_embd_head_k         = 64
print_info: n_embd_head_v         = 64
print_info: n_gqa                 = 7
print_info: n_embd_k_gqa          = 128
print_info: n_embd_v_gqa          = 128
print_info: f_norm_eps            = 0.0e+00
print_info: f_norm_rms_eps        = 1.0e-06
print_info: f_clamp_kqv           = 0.0e+00
print_info: f_max_alibi_bias      = 0.0e+00
print_info: f_logit_scale         = 0.0e+00
print_info: f_attn_scale          = 0.0e+00
print_info: n_ff                  = 4864
print_info: n_expert              = 0
print_info: n_expert_used         = 0
print_info: n_expert_groups       = 0
print_info: n_group_used          = 0
print_info: causal attn           = 1
print_info: pooling type          = -1
print_info: rope type             = 2
print_info: rope scaling          = linear
print_info: freq_base_train       = 1000000.0
print_info: freq_scale_train      = 1
print_info: n_ctx_orig_yarn       = 32768
print_info: rope_yarn_log_mul     = 0.0000
print_info: rope_finetuned        = unknown
print_info: model type            = 1B
print_info: model params          = 630.17 M
print_info: general.name          = qwen2.5-0.5b-instruct
print_info: vocab type            = BPE
print_info: n_vocab               = 151936
print_info: n_merges              = 151387
print_info: BOS token             = 151643 '<|endoftext|>'
print_info: EOS token             = 151645 '<|im_end|>'
print_info: EOT token             = 151645 '<|im_end|>'
print_info: PAD token             = 151643 '<|endoftext|>'
print_info: LF token              = 198 'Ċ'
print_info: FIM PRE token         = 151659 '<|fim_prefix|>'
print_info: FIM SUF token         = 151661 '<|fim_suffix|>'
print_info: FIM MID token         = 151660 '<|fim_middle|>'
print_info: FIM PAD token         = 151662 '<|fim_pad|>'
print_info: FIM REP token         = 151663 '<|repo_name|>'
print_info: FIM SEP token         = 151664 '<|file_sep|>'
print_info: EOG token             = 151643 '<|endoftext|>'
print_info: EOG token             = 151645 '<|im_end|>'
print_info: EOG token             = 151662 '<|fim_pad|>'
print_info: EOG token             = 151663 '<|repo_name|>'
print_info: EOG token             = 151664 '<|file_sep|>'
print_info: max token length      = 256
load_tensors: loading model tensors, this can take a while... (mmap = true, direct_io = false)
load_tensors: layer   0 assigned to device MTL0, is_swa = 0
load_tensors: layer   1 assigned to device MTL0, is_swa = 0
load_tensors: layer   2 assigned to device MTL0, is_swa = 0
load_tensors: layer   3 assigned to device MTL0, is_swa = 0
load_tensors: layer   4 assigned to device MTL0, is_swa = 0
load_tensors: layer   5 assigned to device MTL0, is_swa = 0
load_tensors: layer   6 assigned to device MTL0, is_swa = 0
load_tensors: layer   7 assigned to device MTL0, is_swa = 0
load_tensors: layer   8 assigned to device MTL0, is_swa = 0
load_tensors: layer   9 assigned to device MTL0, is_swa = 0
load_tensors: layer  10 assigned to device MTL0, is_swa = 0
load_tensors: layer  11 assigned to device MTL0, is_swa = 0
load_tensors: layer  12 assigned to device MTL0, is_swa = 0
load_tensors: layer  13 assigned to device MTL0, is_swa = 0
load_tensors: layer  14 assigned to device MTL0, is_swa = 0
load_tensors: layer  15 assigned to device MTL0, is_swa = 0
load_tensors: layer  16 assigned to device MTL0, is_swa = 0
load_tensors: layer  17 assigned to device MTL0, is_swa = 0
load_tensors: layer  18 assigned to device MTL0, is_swa = 0
load_tensors: layer  19 assigned to device MTL0, is_swa = 0
load_tensors: layer  20 assigned to device MTL0, is_swa = 0
load_tensors: layer  21 assigned to device MTL0, is_swa = 0
load_tensors: layer  22 assigned to device MTL0, is_swa = 0
load_tensors: layer  23 assigned to device MTL0, is_swa = 0
load_tensors: layer  24 assigned to device MTL0, is_swa = 0
create_tensor: loading tensor token_embd.weight
create_tensor: loading tensor output_norm.weight
create_tensor: loading tensor output.weight
create_tensor: loading tensor blk.0.attn_norm.weight
create_tensor: loading tensor blk.0.attn_q.weight
create_tensor: loading tensor blk.0.attn_k.weight
create_tensor: loading tensor blk.0.attn_v.weight
create_tensor: loading tensor blk.0.attn_output.weight
create_tensor: loading tensor blk.0.attn_q.bias
create_tensor: loading tensor blk.0.attn_k.bias
create_tensor: loading tensor blk.0.attn_v.bias
create_tensor: loading tensor blk.0.ffn_norm.weight
create_tensor: loading tensor blk.0.ffn_gate.weight
create_tensor: loading tensor blk.0.ffn_down.weight
create_tensor: loading tensor blk.0.ffn_up.weight
create_tensor: loading tensor blk.1.attn_norm.weight
create_tensor: loading tensor blk.1.attn_q.weight
create_tensor: loading tensor blk.1.attn_k.weight
create_tensor: loading tensor blk.1.attn_v.weight
create_tensor: loading tensor blk.1.attn_output.weight
create_tensor: loading tensor blk.1.attn_q.bias
create_tensor: loading tensor blk.1.attn_k.bias
create_tensor: loading tensor blk.1.attn_v.bias
create_tensor: loading tensor blk.1.ffn_norm.weight
create_tensor: loading tensor blk.1.ffn_gate.weight
create_tensor: loading tensor blk.1.ffn_down.weight
create_tensor: loading tensor blk.1.ffn_up.weight
create_tensor: loading tensor blk.2.attn_norm.weight
create_tensor: loading tensor blk.2.attn_q.weight
create_tensor: loading tensor blk.2.attn_k.weight
create_tensor: loading tensor blk.2.attn_v.weight
create_tensor: loading tensor blk.2.attn_output.weight
create_tensor: loading tensor blk.2.attn_q.bias
create_tensor: loading tensor blk.2.attn_k.bias
create_tensor: loading tensor blk.2.attn_v.bias
create_tensor: loading tensor blk.2.ffn_norm.weight
create_tensor: loading tensor blk.2.ffn_gate.weight
create_tensor: loading tensor blk.2.ffn_down.weight
create_tensor: loading tensor blk.2.ffn_up.weight
create_tensor: loading tensor blk.3.attn_norm.weight
create_tensor: loading tensor blk.3.attn_q.weight
create_tensor: loading tensor blk.3.attn_k.weight
create_tensor: loading tensor blk.3.attn_v.weight
create_tensor: loading tensor blk.3.attn_output.weight
create_tensor: loading tensor blk.3.attn_q.bias
create_tensor: loading tensor blk.3.attn_k.bias
create_tensor: loading tensor blk.3.attn_v.bias
create_tensor: loading tensor blk.3.ffn_norm.weight
create_tensor: loading tensor blk.3.ffn_gate.weight
create_tensor: loading tensor blk.3.ffn_down.weight
create_tensor: loading tensor blk.3.ffn_up.weight
create_tensor: loading tensor blk.4.attn_norm.weight
create_tensor: loading tensor blk.4.attn_q.weight
create_tensor: loading tensor blk.4.attn_k.weight
create_tensor: loading tensor blk.4.attn_v.weight
create_tensor: loading tensor blk.4.attn_output.weight
create_tensor: loading tensor blk.4.attn_q.bias
create_tensor: loading tensor blk.4.attn_k.bias
create_tensor: loading tensor blk.4.attn_v.bias
create_tensor: loading tensor blk.4.ffn_norm.weight
create_tensor: loading tensor blk.4.ffn_gate.weight
create_tensor: loading tensor blk.4.ffn_down.weight
create_tensor: loading tensor blk.4.ffn_up.weight
create_tensor: loading tensor blk.5.attn_norm.weight
create_tensor: loading tensor blk.5.attn_q.weight
create_tensor: loading tensor blk.5.attn_k.weight
create_tensor: loading tensor blk.5.attn_v.weight
create_tensor: loading tensor blk.5.attn_output.weight
create_tensor: loading tensor blk.5.attn_q.bias
create_tensor: loading tensor blk.5.attn_k.bias
create_tensor: loading tensor blk.5.attn_v.bias
create_tensor: loading tensor blk.5.ffn_norm.weight
create_tensor: loading tensor blk.5.ffn_gate.weight
create_tensor: loading tensor blk.5.ffn_down.weight
create_tensor: loading tensor blk.5.ffn_up.weight
create_tensor: loading tensor blk.6.attn_norm.weight
create_tensor: loading tensor blk.6.attn_q.weight
create_tensor: loading tensor blk.6.attn_k.weight
create_tensor: loading tensor blk.6.attn_v.weight
create_tensor: loading tensor blk.6.attn_output.weight
create_tensor: loading tensor blk.6.attn_q.bias
create_tensor: loading tensor blk.6.attn_k.bias
create_tensor: loading tensor blk.6.attn_v.bias
create_tensor: loading tensor blk.6.ffn_norm.weight
create_tensor: loading tensor blk.6.ffn_gate.weight
create_tensor: loading tensor blk.6.ffn_down.weight
create_tensor: loading tensor blk.6.ffn_up.weight
create_tensor: loading tensor blk.7.attn_norm.weight
create_tensor: loading tensor blk.7.attn_q.weight
create_tensor: loading tensor blk.7.attn_k.weight
create_tensor: loading tensor blk.7.attn_v.weight
create_tensor: loading tensor blk.7.attn_output.weight
create_tensor: loading tensor blk.7.attn_q.bias
create_tensor: loading tensor blk.7.attn_k.bias
create_tensor: loading tensor blk.7.attn_v.bias
create_tensor: loading tensor blk.7.ffn_norm.weight
create_tensor: loading tensor blk.7.ffn_gate.weight
create_tensor: loading tensor blk.7.ffn_down.weight
create_tensor: loading tensor blk.7.ffn_up.weight
create_tensor: loading tensor blk.8.attn_norm.weight
create_tensor: loading tensor blk.8.attn_q.weight
create_tensor: loading tensor blk.8.attn_k.weight
create_tensor: loading tensor blk.8.attn_v.weight
create_tensor: loading tensor blk.8.attn_output.weight
create_tensor: loading tensor blk.8.attn_q.bias
create_tensor: loading tensor blk.8.attn_k.bias
create_tensor: loading tensor blk.8.attn_v.bias
create_tensor: loading tensor blk.8.ffn_norm.weight
create_tensor: loading tensor blk.8.ffn_gate.weight
create_tensor: loading tensor blk.8.ffn_down.weight
create_tensor: loading tensor blk.8.ffn_up.weight
create_tensor: loading tensor blk.9.attn_norm.weight
create_tensor: loading tensor blk.9.attn_q.weight
create_tensor: loading tensor blk.9.attn_k.weight
create_tensor: loading tensor blk.9.attn_v.weight
create_tensor: loading tensor blk.9.attn_output.weight
create_tensor: loading tensor blk.9.attn_q.bias
create_tensor: loading tensor blk.9.attn_k.bias
create_tensor: loading tensor blk.9.attn_v.bias
create_tensor: loading tensor blk.9.ffn_norm.weight
create_tensor: loading tensor blk.9.ffn_gate.weight
create_tensor: loading tensor blk.9.ffn_down.weight
create_tensor: loading tensor blk.9.ffn_up.weight
create_tensor: loading tensor blk.10.attn_norm.weight
create_tensor: loading tensor blk.10.attn_q.weight
create_tensor: loading tensor blk.10.attn_k.weight
create_tensor: loading tensor blk.10.attn_v.weight
create_tensor: loading tensor blk.10.attn_output.weight
create_tensor: loading tensor blk.10.attn_q.bias
create_tensor: loading tensor blk.10.attn_k.bias
create_tensor: loading tensor blk.10.attn_v.bias
create_tensor: loading tensor blk.10.ffn_norm.weight
create_tensor: loading tensor blk.10.ffn_gate.weight
create_tensor: loading tensor blk.10.ffn_down.weight
create_tensor: loading tensor blk.10.ffn_up.weight
create_tensor: loading tensor blk.11.attn_norm.weight
create_tensor: loading tensor blk.11.attn_q.weight
create_tensor: loading tensor blk.11.attn_k.weight
create_tensor: loading tensor blk.11.attn_v.weight
create_tensor: loading tensor blk.11.attn_output.weight
create_tensor: loading tensor blk.11.attn_q.bias
create_tensor: loading tensor blk.11.attn_k.bias
create_tensor: loading tensor blk.11.attn_v.bias
create_tensor: loading tensor blk.11.ffn_norm.weight
create_tensor: loading tensor blk.11.ffn_gate.weight
create_tensor: loading tensor blk.11.ffn_down.weight
create_tensor: loading tensor blk.11.ffn_up.weight
create_tensor: loading tensor blk.12.attn_norm.weight
create_tensor: loading tensor blk.12.attn_q.weight
create_tensor: loading tensor blk.12.attn_k.weight
create_tensor: loading tensor blk.12.attn_v.weight
create_tensor: loading tensor blk.12.attn_output.weight
create_tensor: loading tensor blk.12.attn_q.bias
create_tensor: loading tensor blk.12.attn_k.bias
create_tensor: loading tensor blk.12.attn_v.bias
create_tensor: loading tensor blk.12.ffn_norm.weight
create_tensor: loading tensor blk.12.ffn_gate.weight
create_tensor: loading tensor blk.12.ffn_down.weight
create_tensor: loading tensor blk.12.ffn_up.weight
create_tensor: loading tensor blk.13.attn_norm.weight
create_tensor: loading tensor blk.13.attn_q.weight
create_tensor: loading tensor blk.13.attn_k.weight
create_tensor: loading tensor blk.13.attn_v.weight
create_tensor: loading tensor blk.13.attn_output.weight
create_tensor: loading tensor blk.13.attn_q.bias
create_tensor: loading tensor blk.13.attn_k.bias
create_tensor: loading tensor blk.13.attn_v.bias
create_tensor: loading tensor blk.13.ffn_norm.weight
create_tensor: loading tensor blk.13.ffn_gate.weight
create_tensor: loading tensor blk.13.ffn_down.weight
create_tensor: loading tensor blk.13.ffn_up.weight
create_tensor: loading tensor blk.14.attn_norm.weight
create_tensor: loading tensor blk.14.attn_q.weight
create_tensor: loading tensor blk.14.attn_k.weight
create_tensor: loading tensor blk.14.attn_v.weight
create_tensor: loading tensor blk.14.attn_output.weight
create_tensor: loading tensor blk.14.attn_q.bias
create_tensor: loading tensor blk.14.attn_k.bias
create_tensor: loading tensor blk.14.attn_v.bias
create_tensor: loading tensor blk.14.ffn_norm.weight
create_tensor: loading tensor blk.14.ffn_gate.weight
create_tensor: loading tensor blk.14.ffn_down.weight
create_tensor: loading tensor blk.14.ffn_up.weight
create_tensor: loading tensor blk.15.attn_norm.weight
create_tensor: loading tensor blk.15.attn_q.weight
create_tensor: loading tensor blk.15.attn_k.weight
create_tensor: loading tensor blk.15.attn_v.weight
create_tensor: loading tensor blk.15.attn_output.weight
create_tensor: loading tensor blk.15.attn_q.bias
create_tensor: loading tensor blk.15.attn_k.bias
create_tensor: loading tensor blk.15.attn_v.bias
create_tensor: loading tensor blk.15.ffn_norm.weight
create_tensor: loading tensor blk.15.ffn_gate.weight
create_tensor: loading tensor blk.15.ffn_down.weight
create_tensor: loading tensor blk.15.ffn_up.weight
create_tensor: loading tensor blk.16.attn_norm.weight
create_tensor: loading tensor blk.16.attn_q.weight
create_tensor: loading tensor blk.16.attn_k.weight
create_tensor: loading tensor blk.16.attn_v.weight
create_tensor: loading tensor blk.16.attn_output.weight
create_tensor: loading tensor blk.16.attn_q.bias
create_tensor: loading tensor blk.16.attn_k.bias
create_tensor: loading tensor blk.16.attn_v.bias
create_tensor: loading tensor blk.16.ffn_norm.weight
create_tensor: loading tensor blk.16.ffn_gate.weight
create_tensor: loading tensor blk.16.ffn_down.weight
create_tensor: loading tensor blk.16.ffn_up.weight
create_tensor: loading tensor blk.17.attn_norm.weight
create_tensor: loading tensor blk.17.attn_q.weight
create_tensor: loading tensor blk.17.attn_k.weight
create_tensor: loading tensor blk.17.attn_v.weight
create_tensor: loading tensor blk.17.attn_output.weight
create_tensor: loading tensor blk.17.attn_q.bias
create_tensor: loading tensor blk.17.attn_k.bias
create_tensor: loading tensor blk.17.attn_v.bias
create_tensor: loading tensor blk.17.ffn_norm.weight
create_tensor: loading tensor blk.17.ffn_gate.weight
create_tensor: loading tensor blk.17.ffn_down.weight
create_tensor: loading tensor blk.17.ffn_up.weight
create_tensor: loading tensor blk.18.attn_norm.weight
create_tensor: loading tensor blk.18.attn_q.weight
create_tensor: loading tensor blk.18.attn_k.weight
create_tensor: loading tensor blk.18.attn_v.weight
create_tensor: loading tensor blk.18.attn_output.weight
create_tensor: loading tensor blk.18.attn_q.bias
create_tensor: loading tensor blk.18.attn_k.bias
create_tensor: loading tensor blk.18.attn_v.bias
create_tensor: loading tensor blk.18.ffn_norm.weight
create_tensor: loading tensor blk.18.ffn_gate.weight
create_tensor: loading tensor blk.18.ffn_down.weight
create_tensor: loading tensor blk.18.ffn_up.weight
create_tensor: loading tensor blk.19.attn_norm.weight
create_tensor: loading tensor blk.19.attn_q.weight
create_tensor: loading tensor blk.19.attn_k.weight
create_tensor: loading tensor blk.19.attn_v.weight
create_tensor: loading tensor blk.19.attn_output.weight
create_tensor: loading tensor blk.19.attn_q.bias
create_tensor: loading tensor blk.19.attn_k.bias
create_tensor: loading tensor blk.19.attn_v.bias
create_tensor: loading tensor blk.19.ffn_norm.weight
create_tensor: loading tensor blk.19.ffn_gate.weight
create_tensor: loading tensor blk.19.ffn_down.weight
create_tensor: loading tensor blk.19.ffn_up.weight
create_tensor: loading tensor blk.20.attn_norm.weight
create_tensor: loading tensor blk.20.attn_q.weight
create_tensor: loading tensor blk.20.attn_k.weight
create_tensor: loading tensor blk.20.attn_v.weight
create_tensor: loading tensor blk.20.attn_output.weight
create_tensor: loading tensor blk.20.attn_q.bias
create_tensor: loading tensor blk.20.attn_k.bias
create_tensor: loading tensor blk.20.attn_v.bias
create_tensor: loading tensor blk.20.ffn_norm.weight
create_tensor: loading tensor blk.20.ffn_gate.weight
create_tensor: loading tensor blk.20.ffn_down.weight
create_tensor: loading tensor blk.20.ffn_up.weight
create_tensor: loading tensor blk.21.attn_norm.weight
create_tensor: loading tensor blk.21.attn_q.weight
create_tensor: loading tensor blk.21.attn_k.weight
create_tensor: loading tensor blk.21.attn_v.weight
create_tensor: loading tensor blk.21.attn_output.weight
create_tensor: loading tensor blk.21.attn_q.bias
create_tensor: loading tensor blk.21.attn_k.bias
create_tensor: loading tensor blk.21.attn_v.bias
create_tensor: loading tensor blk.21.ffn_norm.weight
create_tensor: loading tensor blk.21.ffn_gate.weight
create_tensor: loading tensor blk.21.ffn_down.weight
create_tensor: loading tensor blk.21.ffn_up.weight
create_tensor: loading tensor blk.22.attn_norm.weight
create_tensor: loading tensor blk.22.attn_q.weight
create_tensor: loading tensor blk.22.attn_k.weight
create_tensor: loading tensor blk.22.attn_v.weight
create_tensor: loading tensor blk.22.attn_output.weight
create_tensor: loading tensor blk.22.attn_q.bias
create_tensor: loading tensor blk.22.attn_k.bias
create_tensor: loading tensor blk.22.attn_v.bias
create_tensor: loading tensor blk.22.ffn_norm.weight
create_tensor: loading tensor blk.22.ffn_gate.weight
create_tensor: loading tensor blk.22.ffn_down.weight
create_tensor: loading tensor blk.22.ffn_up.weight
create_tensor: loading tensor blk.23.attn_norm.weight
create_tensor: loading tensor blk.23.attn_q.weight
create_tensor: loading tensor blk.23.attn_k.weight
create_tensor: loading tensor blk.23.attn_v.weight
create_tensor: loading tensor blk.23.attn_output.weight
create_tensor: loading tensor blk.23.attn_q.bias
create_tensor: loading tensor blk.23.attn_k.bias
create_tensor: loading tensor blk.23.attn_v.bias
create_tensor: loading tensor blk.23.ffn_norm.weight
create_tensor: loading tensor blk.23.ffn_gate.weight
create_tensor: loading tensor blk.23.ffn_down.weight
create_tensor: loading tensor blk.23.ffn_up.weight
load_tensors: tensor 'token_embd.weight' (q5_0) (and 0 others) cannot be used with preferred buffer type CPU_REPACK, using CPU instead
ggml_metal_log_allocated_size: allocated buffer, size =   462.97 MiB, (  463.42 / 12124.17)
load_tensors: offloading output layer to GPU
load_tensors: offloading 23 repeating layers to GPU
load_tensors: offloaded 25/25 layers to GPU
load_tensors:   CPU_Mapped model buffer size =    89.26 MiB
load_tensors:  MTL0_Mapped model buffer size =   462.96 MiB
.....................................................
llama_context: constructing llama_context
llama_context: n_seq_max     = 1
llama_context: n_ctx         = 512
llama_context: n_ctx_seq     = 512
llama_context: n_batch       = 512
llama_context: n_ubatch      = 512
llama_context: causal_attn   = 1
llama_context: flash_attn    = auto
llama_context: kv_unified    = false
llama_context: freq_base     = 1000000.0
llama_context: freq_scale    = 1
llama_context: n_ctx_seq (512) < n_ctx_train (32768) -- the full capacity of the model will not be utilized
ggml_metal_init: allocating
ggml_metal_init: found device: Apple M4
ggml_metal_init: picking default device: Apple M4
ggml_metal_init: use fusion         = true
ggml_metal_init: use concurrency    = true
ggml_metal_init: use graph optimize = true
set_abort_callback: call
llama_context:        CPU  output buffer size =     0.58 MiB
llama_kv_cache: layer   0: dev = MTL0
llama_kv_cache: layer   1: dev = MTL0
llama_kv_cache: layer   2: dev = MTL0
llama_kv_cache: layer   3: dev = MTL0
llama_kv_cache: layer   4: dev = MTL0
llama_kv_cache: layer   5: dev = MTL0
llama_kv_cache: layer   6: dev = MTL0
llama_kv_cache: layer   7: dev = MTL0
llama_kv_cache: layer   8: dev = MTL0
llama_kv_cache: layer   9: dev = MTL0
llama_kv_cache: layer  10: dev = MTL0
llama_kv_cache: layer  11: dev = MTL0
llama_kv_cache: layer  12: dev = MTL0
llama_kv_cache: layer  13: dev = MTL0
llama_kv_cache: layer  14: dev = MTL0
llama_kv_cache: layer  15: dev = MTL0
llama_kv_cache: layer  16: dev = MTL0
llama_kv_cache: layer  17: dev = MTL0
llama_kv_cache: layer  18: dev = MTL0
llama_kv_cache: layer  19: dev = MTL0
llama_kv_cache: layer  20: dev = MTL0
llama_kv_cache: layer  21: dev = MTL0
llama_kv_cache: layer  22: dev = MTL0
llama_kv_cache: layer  23: dev = MTL0
llama_kv_cache:       MTL0 KV buffer size =     6.00 MiB
llama_kv_cache: size =    6.00 MiB (   512 cells,  24 layers,  1/1 seqs), K (f16):    3.00 MiB, V (f16):    3.00 MiB
llama_context: enumerating backends
llama_context: backend_ptrs.size() = 3
sched_reserve: reserving ...
sched_reserve: max_nodes = 2328
sched_reserve: reserving full memory module
sched_reserve: worst-case: n_tokens = 512, n_seqs = 1, n_outputs = 1
graph_reserve: reserving a graph for ubatch with n_tokens =    1, n_seqs =  1, n_outputs =    1
sched_reserve: Flash Attention was auto, set to enabled
graph_reserve: reserving a graph for ubatch with n_tokens =  512, n_seqs =  1, n_outputs =  512
graph_reserve: reserving a graph for ubatch with n_tokens =    1, n_seqs =  1, n_outputs =    1
graph_reserve: reserving a graph for ubatch with n_tokens =  512, n_seqs =  1, n_outputs =  512
sched_reserve:       MTL0 compute buffer size =   300.25 MiB
sched_reserve:        CPU compute buffer size =     4.51 MiB
sched_reserve: graph nodes  = 823
sched_reserve: graph splits = 2
sched_reserve: reserve took 66.73 ms, sched copies = 1
ggml_metal_library_compile_pipeline: compiling pipeline: base = 'kernel_rms_norm_mul_f32_4', name = 'kernel_rms_norm_mul_f32_4'
ggml_metal_library_compile_pipeline: loaded kernel_rms_norm_mul_f32_4                     0x1030e8e10 | th_max = 1024 | th_width =   32
ggml_metal_library_compile_pipeline: compiling pipeline: base = 'kernel_mul_mv_q5_0_f32', name = 'kernel_mul_mv_q5_0_f32_nsg=2'
ggml_metal_library_compile_pipeline: loaded kernel_mul_mv_q5_0_f32_nsg=2                  0x1030e9910 | th_max = 1024 | th_width =   32
ggml_metal_library_compile_pipeline: compiling pipeline: base = 'kernel_mul_mv_q8_0_f32', name = 'kernel_mul_mv_q8_0_f32_nsg=4'
ggml_metal_library_compile_pipeline: loaded kernel_mul_mv_q8_0_f32_nsg=4                  0x1030ea410 | th_max = 1024 | th_width =   32
ggml_metal_library_compile_pipeline: compiling pipeline: base = 'kernel_add_row_c4_fuse_1', name = 'kernel_add_row_c4_fuse_1'
ggml_metal_library_compile_pipeline: loaded kernel_add_row_c4_fuse_1                      0x1030eac10 | th_max = 1024 | th_width =   32
ggml_metal_library_compile_pipeline: compiling pipeline: base = 'kernel_rope_neox_f32', name = 'kernel_rope_neox_f32_imrope=0'
ggml_metal_library_compile_pipeline: loaded kernel_rope_neox_f32_imrope=0                 0x1030eaf10 | th_max = 1024 | th_width =   32
ggml_metal_library_compile_pipeline: compiling pipeline: base = 'kernel_set_rows_f16_i64', name = 'kernel_set_rows_f16_i64'
ggml_metal_library_compile_pipeline: loaded kernel_set_rows_f16_i64                       0x7baef8000 | th_max = 1024 | th_width =   32
ggml_metal_library_compile_pipeline: compiling pipeline: base = 'kernel_cpy_f32_f16', name = 'kernel_cpy_f32_f16'
ggml_metal_library_compile_pipeline: loaded kernel_cpy_f32_f16                            0x7baef8300 | th_max = 1024 | th_width =   32
ggml_metal_library_compile_pipeline: compiling pipeline: base = 'kernel_flash_attn_ext_vec_f16_dk64_dv64', name = 'kernel_flash_attn_ext_vec_f16_dk64_dv64_mask=1_sink=0_bias=0_scap=0_kvpad=0_ns10=128_ns20=128_nsg=1_nwg=32'
ggml_metal_library_compile_pipeline: loaded kernel_flash_attn_ext_vec_f16_dk64_dv64_mask=1_sink=0_bias=0_scap=0_kvpad=0_ns10=128_ns20=128_nsg=1_nwg=32      0x7baef8600 | th_max = 1024 | th_width =   32
ggml_metal_library_compile_pipeline: compiling pipeline: base = 'kernel_flash_attn_ext_vec_reduce', name = 'kernel_flash_attn_ext_vec_reduce_dv=64_nwg=32'
ggml_metal_library_compile_pipeline: loaded kernel_flash_attn_ext_vec_reduce_dv=64_nwg=32      0x7baef8900 | th_max = 1024 | th_width =   32
ggml_metal_library_compile_pipeline: compiling pipeline: base = 'kernel_swiglu_f32', name = 'kernel_swiglu_f32'
ggml_metal_library_compile_pipeline: loaded kernel_swiglu_f32                             0x7baef8c00 | th_max = 1024 | th_width =   32
ggml_metal_library_compile_pipeline: compiling pipeline: base = 'kernel_mul_mv_q6_K_f32', name = 'kernel_mul_mv_q6_K_f32_nsg=2'
ggml_metal_library_compile_pipeline: loaded kernel_mul_mv_q6_K_f32_nsg=2                  0x7baef8f00 | th_max = 1024 | th_width =   32
ggml_metal_library_compile_pipeline: compiling pipeline: base = 'kernel_mul_mv_q4_K_f32', name = 'kernel_mul_mv_q4_K_f32_nsg=2'
ggml_metal_library_compile_pipeline: loaded kernel_mul_mv_q4_K_f32_nsg=2                  0x7baef9200 | th_max = 1024 | th_width =   32
ggml_metal_library_compile_pipeline: compiling pipeline: base = 'kernel_get_rows_f32', name = 'kernel_get_rows_f32'
ggml_metal_library_compile_pipeline: loaded kernel_get_rows_f32                           0x7baef9500 | th_max = 1024 | th_width =   32



I am trying to create a simple program that will take a string and a number as input, and then output the string with the number of characters removed. Can someone help me with this?

```python
def remove_chars(s, n):
    return s[:n]
```

I want to test this function with the following inputs:
- `remove_chars("Hello", 5)` should return `"Hello"`
- `remove_chars("Hello", 10)` should return `"Hello"`
- `remove_chars("Hello", 1)` should return `"ello"`

Can someone provide a solution for this? ```python
def remove_chars

[Duration: 0 ms, Tokens: 128]
~llama_context:       MTL0 compute buffer size is 300.2500 MiB, matches expectation of 300.2500 MiB
~llama_context:        CPU compute buffer size is   4.5137 MiB, matches expectation of   4.5137 MiB
~llama_context:        CPU compute buffer size is   0.0000 MiB, matches expectation of   0.0000 MiB
ggml_metal_free: deallocating