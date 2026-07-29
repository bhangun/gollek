import os

dir_path = "/Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/gollek/spi/gollek-spi-model"
for root, dirs, files in os.walk(dir_path):
    for file in files:
        if file.endswith(".java"):
            file_path = os.path.join(root, file)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            old_import = "import tech.kayys.gollek.core.model.ModelFormat;"
            new_import = "import tech.kayys.alkhawarizm.core.model.ModelFormat;"
            
            if old_import in content:
                print(f"Fixing imports in {file_path}")
                content = content.replace(old_import, new_import)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
