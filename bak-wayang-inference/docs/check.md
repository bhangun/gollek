

 context-related functions exist:

```bash
nm -g build/bin/libllama.dylib | grep -E "(context|free)" | grep llama

```

Check what functions actually exist

 see what free/destructor functions are available:

```bash
nm -g build/bin/libllama.dylib | grep -E "(free|destroy|delete)" | grep llama
```