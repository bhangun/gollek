# Gollek SDK - Jupyter & jbang Support

**Status**: Setup for easy interactive development with Jupyter notebooks and jbang scripts.

## Quick Start

### Option 1: Jupyter Notebooks (Interactive GUI)

```bash
# Install Jupyter Java kernel
jupyter kernelspec install --user /path/to/gollek/sdk/jupyter-kernel

# Start Jupyter
jupyter notebook

# Create new notebook with "Gollek SDK" kernel
# Then use gollek components directly:
```

```java
import tech.kayys.gollek.ml.nn.*;
import tech.kayys.gollek.ml.autograd.*;

// Build your neural network
Module model = new Sequential(
    new Linear(784, 256),
    new ReLU(),
    new Linear(256, 10)
);
```

### Option 2: jbang (CLI + Script Execution)

```bash
# Install jbang alias
jbang alias add gollek https://github.com/your-repo/gollek-sdk

# Run jbang script
jbang gollek-example.java

# Or use inline
jbang --version
```

## Documentation

- **[JUPYTER_SETUP.md](JUPYTER_SETUP.md)** - Detailed Jupyter installation and usage
- **[JBANG_SETUP.md](JBANG_SETUP.md)** - jbang configuration and scripting
- **[EXAMPLES.md](EXAMPLES.md)** - Example notebooks and scripts
- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - Common issues and solutions

## Structure

```
gollek/sdk/
├── jupyter-kernel/          # Jupyter kernel configuration
│   ├── kernel.json
│   └── install.sh
├── jbang-templates/         # jbang script templates
│   ├── template.java
│   └── examples/
├── examples/                # Notebook and script examples
│   ├── notebooks/
│   └── scripts/
└── [documentation files]
```

## Features

✅ Full gollek SDK available in Jupyter notebooks
✅ Interactive development with instant feedback
✅ jbang scripts for automation
✅ All modules (nn, data, ml, nlp, etc.)
✅ Proper classpath management
✅ Example notebooks and scripts included

## Support

For questions or issues:
1. Check TROUBLESHOOTING.md
2. Review example notebooks/scripts
3. Check kernel configuration

See individual documentation files for detailed setup and usage instructions.
