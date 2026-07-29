import os

dir_path = "/Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/gollek/runner/gguf/gollek-gguf-core"

replacements = {
    "import tech.kayys.gollek.ml.autograd.GradTensor;": "import tech.kayys.alkhawarizm.ml.autograd.GradTensor;",
    "import tech.kayys.gollek.ml.nn.NNModule;": "import tech.kayys.alkhawarizm.nn.NNModule;",
    "import tech.kayys.gollek.tokenizer.impl.BpeTokenizer;": "import tech.kayys.alkhawarizm.tokenizer.impl.BpeTokenizer;",
    "import tech.kayys.gollek.tokenizer.impl.Gpt2PreTokenizer;": "import tech.kayys.alkhawarizm.tokenizer.impl.Gpt2PreTokenizer;",
    "import tech.kayys.gollek.tokenizer.spi.DecodeOptions;": "import tech.kayys.alkhawarizm.tokenizer.spi.DecodeOptions;",
    "import tech.kayys.gollek.tokenizer.spi.EncodeOptions;": "import tech.kayys.alkhawarizm.tokenizer.spi.EncodeOptions;",
    "import tech.kayys.gollek.tokenizer.spi.Tokenizer;": "import tech.kayys.alkhawarizm.tokenizer.spi.Tokenizer;"
}

for root, dirs, files in os.walk(dir_path):
    for file in files:
        if file.endswith(".java"):
            file_path = os.path.join(root, file)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            modified = False
            for old, new in replacements.items():
                if old in content:
                    content = content.replace(old, new)
                    modified = True
            
            if modified:
                print(f"Fixing imports in {file_path}")
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
