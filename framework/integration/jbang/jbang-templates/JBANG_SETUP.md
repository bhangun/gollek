# jbang Setup Guide - Gollek SDK

Complete guide for setting up and using Gollek SDK with jbang scripts.

## Table of Contents
1. [What is jbang?](#what-is-jbang)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Scripting](#scripting)
5. [Examples](#examples)
6. [Troubleshooting](#troubleshooting)

## What is jbang?

**jbang** is a tool for writing and executing Java scripts with zero boilerplate:
- Single-file Java scripts (like Python!)
- Automatic dependency resolution from Maven
- No compilation step needed
- Perfect for automation, CLI tools, and scripting

## Installation

### Step 1: Install jbang

**macOS/Linux:**
```bash
curl -Ls https://sh.jbang.dev | bash -s - app setup
```

**Windows (using scoop):**
```powershell
scoop install jbang
```

**Or manually:**
```bash
# Download and extract from GitHub releases
# https://github.com/jbangdev/jbang/releases
```

### Step 2: Verify Installation

```bash
jbang --version
```

Should output something like: `jbang 0.x.x`

### Step 3: Setup Gollek SDK Aliases (Optional)

Create `~/.jbang/jbang.properties` to configure aliases:

```properties
# Gollek SDK alias
aliases.gollek=file://.../gollek/sdk/jbang-templates/gollek-template.java
```

Or use direct URL if published:
```properties
aliases.gollek=https://github.com/your-repo/gollek-sdk/jbang-templates/gollek-template.java
```

## Quick Start

### 1. Create Your First jbang Script

```java
///usr/bin/env jbang
// DEPS tech.kayys:gollek-sdk-nn:1.0.0

public class HelloGollek {
    public static void main(String[] args) {
        System.out.println("🚀 Hello from Gollek SDK!");
    }
}
```

Save as `hello.java`

### 2. Run It

```bash
jbang hello.java
# Output: 🚀 Hello from Gollek SDK!
```

That's it! No compilation, no JAR files, no boilerplate.

### 3. Accept Arguments

```java
///usr/bin/env jbang

public class HelloGollek {
    public static void main(String[] args) {
        if (args.length > 0) {
            System.out.println("Hello " + args[0] + "!");
        } else {
            System.out.println("Hello Gollek!");
        }
    }
}
```

```bash
jbang hello.java Alice
# Output: Hello Alice!
```

## Scripting

### Basic Script Structure

```java
///usr/bin/env jbang
// DEPS dependency:1.0.0
// SOURCES other_file.java
// COMPILE_OPTIONS --enable-preview

import some.package.*;

public class MyScript {
    public static void main(String[] args) {
        // Your code here
    }
}
```

### Adding Dependencies

```java
// Single dependency
// DEPS org.slf4j:slf4j-simple:2.0.0

// Multiple dependencies
// DEPS tech.kayys:gollek-sdk-nn:1.0.0
// DEPS tech.kayys:gollek-sdk-autograd:1.0.0
// DEPS org.apache.commons:commons-lang3:3.12.0
```

### With Repositories

```java
// Use specific Maven repository
// REPOS https://repo.maven.apache.org/maven2
// DEPS com.example:artifact:1.0.0
```

### Script Options

```java
// Enable Java preview features
// COMPILE_OPTIONS --enable-preview
// RUNTIME_OPTIONS --enable-preview

// Set Java version
// JAVA 17+

// Include other files
// SOURCES Helper.java
// FILES data.csv
```

## Examples

### Example 1: Neural Network Builder

```java
///usr/bin/env jbang
// DEPS tech.kayys:gollek-sdk-nn:1.0.0
// DEPS tech.kayys:gollek-sdk-autograd:1.0.0

import tech.kayys.gollek.ml.nn.*;
import tech.kayys.gollek.ml.autograd.*;

public class NeuralNetBuilder {
    
    static Module buildClassifier(int inputSize, int... hiddenSizes) {
        System.out.println("Building classifier...");
        System.out.println("Input: " + inputSize);
        for (int h : hiddenSizes) System.out.println("Hidden: " + h);
        
        Module model = new Sequential(
            new Linear(inputSize, hiddenSizes[0]),
            new ReLU()
        );
        
        for (int i = 1; i < hiddenSizes.length; i++) {
            // Add more layers
        }
        
        return model;
    }
    
    public static void main(String[] args) {
        Module model = buildClassifier(784, 256, 128, 10);
        System.out.println("✓ Model created!");
    }
}
```

```bash
jbang NeuralNetBuilder.java
```

### Example 2: Batch Data Processor

```java
///usr/bin/env jbang
// DEPS tech.kayys:gollek-sdk-data:1.0.0

import java.nio.file.*;
import java.util.*;

public class BatchProcessor {
    public static void main(String[] args) {
        if (args.length < 1) {
            System.err.println("Usage: batch-processor <directory>");
            System.exit(1);
        }
        
        Path dir = Paths.get(args[0]);
        System.out.println("Processing: " + dir);
        
        try {
            Files.list(dir)
                .filter(p -> p.toString().endsWith(".csv"))
                .forEach(p -> System.out.println("Found: " + p));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
```

```bash
jbang BatchProcessor.java /path/to/data
```

### Example 3: Model Trainer

```java
///usr/bin/env jbang
// DEPS tech.kayys:gollek-sdk-nn:1.0.0
// DEPS tech.kayys:gollek-sdk-autograd:1.0.0

import tech.kayys.gollek.ml.nn.*;
import tech.kayys.gollek.ml.autograd.*;

public class ModelTrainer {
    
    static void trainEpoch(Module model, 
                          tech.kayys.gollek.ml.nn.optim.Optimizer optimizer,
                          tech.kayys.gollek.ml.nn.loss.CrossEntropyLoss loss,
                          float[][] batchData,
                          float[] batchLabels) {
        for (int i = 0; i < batchData.length; i++) {
            // Forward
            var input = GradTensor.of(batchData[i], new long[]{1, batchData[i].length});
            var output = model.forward(input);
            
            // Loss
            var target = GradTensor.of(new float[]{batchLabels[i]}, new long[]{1});
            var lossVal = loss.compute(output, target);
            
            // Backward
            lossVal.backward();
            optimizer.step();
            optimizer.zeroGrad();
        }
    }
    
    public static void main(String[] args) {
        System.out.println("Model Trainer");
        System.out.println("Creating model...");
        
        Module model = new Sequential(
            new Linear(784, 256),
            new ReLU(),
            new Linear(256, 10)
        );
        
        System.out.println("✓ Model ready for training!");
    }
}
```

## Running Templates

### From Local File

```bash
# Using template from jbang-templates directory
jbang jbang-templates/gollek-template.java

# Using example script
jbang jbang-templates/examples/neural-network-example.java
```

### Using Alias

```bash
# After setting up alias
jbang gollek

# Which runs the template script
```

### From GitHub

```bash
# Run directly from URL
jbang https://github.com/your-repo/gollek-sdk/jbang-templates/examples/neural-network-example.java
```

## Advanced Features

### 1. Creating a CLI Tool

```java
///usr/bin/env jbang
// DEPS picocli:picocli:4.6.0

import picocli.CommandLine;
import picocli.CommandLine.*;

@Command(name = "gollek", description = "Gollek SDK CLI")
public class GolleKCLI implements Runnable {
    
    @Option(names = {"-h", "--help"}, usageHelp = true)
    boolean help;
    
    @Command
    void train(@Parameters String modelPath) {
        System.out.println("Training: " + modelPath);
    }
    
    public void run() {
        System.out.println("Gollek SDK");
    }
    
    public static void main(String[] args) {
        System.exit(new CommandLine(new GolleKCLI()).execute(args));
    }
}
```

### 2. Using System Properties

```java
// Inside your script
String version = System.getProperty("gollek.version", "1.0.0");
System.out.println("Version: " + version);
```

```bash
jbang -Dgollek.version=2.0.0 script.java
```

### 3. Edit and Debug

```bash
# Open in editor
jbang --edit MyScript.java

# Run with debug port
jbang --debug MyScript.java
```

## Troubleshooting

### Issue: Dependency not found

**Solution:**
```bash
# Clear local cache
rm -rf ~/.jbang

# Try again (will re-download)
jbang script.java
```

### Issue: Out of memory

**Solution:**
```bash
# Increase heap
jbang -Xmx2G script.java

# Or set permanently in script
// RUNTIME_OPTIONS -Xmx2G
```

### Issue: Cannot find symbol

**Solution:**
- Check dependency spelling
- Verify Maven repository availability
- Check Java version compatibility

```bash
# List resolved dependencies
jbang --info script.java
```

### Issue: Slow first run

**Solution:**
- First run downloads and caches dependencies
- Subsequent runs are fast
- Use `-o` to output JAR for faster re-use

```bash
jbang -o my-app.jar script.java
java -jar my-app.jar
```

## Tips & Best Practices

1. **Start Simple**: Begin with minimal dependencies
2. **Use Latest Versions**: Keep dependencies up to date
3. **Cache Builds**: Save as JAR for production
4. **Document Dependencies**: Clearly list all DEPS
5. **Test Locally**: Test scripts before distribution
6. **Use Aliases**: For frequently used scripts
7. **Version Your Scripts**: Use consistent versions

## Publishing Scripts

### Local Distribution

```bash
# Export as standalone JAR
jbang -o my-script.jar my-script.java

# Share the JAR file
```

### GitHub Distribution

```bash
# Push to GitHub
git push origin main

# Anyone can run with:
jbang https://github.com/user/repo/blob/main/script.java
```

### Maven Central

- Publish your application as Maven artifact
- Reference in jbang scripts:
  ```java
  // DEPS com.example:my-app:1.0.0
  ```

## Next Steps

1. Try the [examples](../jbang-templates/examples/)
2. Create your first script
3. Automate with jbang
4. Share with others!

## Additional Resources

- [jbang Documentation](https://jbang.dev/)
- [jbang GitHub](https://github.com/jbangdev/jbang)
- [Gollek SDK Documentation](../README.md)

## Support

For issues:
1. Check this guide
2. See [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
3. Review jbang documentation
4. Ask community

---

**Happy Scripting with jbang! 🚀**
