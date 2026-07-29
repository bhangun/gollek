import os

dir_path = "/Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/gollek/runner/onnx/gollek-runner-onnx"

replacements = {
    "import tech.kayys.gollek.spi.inference.": "import tech.kayys.alkhawarizm.spi.inference.",
    "import tech.kayys.gollek.spi.model.MultimodalProcessor;": "import tech.kayys.alkhawarizm.spi.model.MultimodalProcessor;",
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
