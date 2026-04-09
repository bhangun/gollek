# Using Gollek SDK with jbang - Setup Guide

## Overview

jbang is a tool for creating single-file Java scripts with automatic dependency resolution. This guide explains how to use Gollek SDK with jbang.

## Quick Start

### 1. Install jbang

```bash
curl -Ls https://sh.jbang.dev | bash -s - app setup
```

Verify installation:
```bash
jbang --version
```

### 2. Run the Template

```bash
# Navigate to gollek/sdk
cd gollek/sdk

# Run the basic template
jbang jbang-templates/gollek_template.java
```

**Expected output:**
```
🚀 Gollek SDK jbang Template
============================
[examples shown...]
✅ Template examples completed successfully!
```

### 3. Try with Arguments

```bash
jbang jbang-templates/gollek_template.java arg1 arg2 arg3
```

The script will echo back your arguments.

---

## Using Gollek SDK Classes in Your Scripts

### Option 1: Local Maven Installation (Recommended)

If you've built Gollek SDK locally with `mvn clean install`, jbang can use it from your local Maven repository:

**1. Build the SDK**
```bash
cd gollek/sdk
mvn clean install -DskipTests
```

**2. Create a jbang script with DEPS**

```java
///usr/bin/env jbang
// DEPS tech.kayys.gollek:gollek-sdk-nn:1.0.0-SNAPSHOT
// DEPS org.slf4j:slf4j-simple:2.0.0

import tech.kayys.gollek.ml.nn.*;

public class my_model {
    public static void main(String[] args) {
        // Create model
        Module model = new Sequential(
            new Linear(784, 256),
            new ReLU(),
            new Linear(256, 10)
        );
        
        System.out.println("Model created: " + model);
    }
}
```

**3. Run the script**
```bash
jbang my_model.java
```

### Option 2: Publishing to Maven Central

To make Gollek SDK available to everyone via Maven Central:

1. Configure Maven pom.xml with Sonatype credentials
2. Run: `mvn clean deploy`
3. Verify on Maven Central
4. Update DEPS with the released version (no SNAPSHOT)

See `pom.xml` for distribution configuration.

---

## Common DEPS Coordinates

### Gollek SDK Modules

```java
// Neural Network Module (includes nn.* classes)
// DEPS tech.kayys.gollek:gollek-sdk-nn:1.0.0-SNAPSHOT

// Data Module (includes data processing)
// DEPS tech.kayys.gollek:gollek-sdk-data:1.0.0-SNAPSHOT

// ML Module (machine learning utilities)
// DEPS tech.kayys.gollek:gollek-sdk-ml:1.0.0-SNAPSHOT

// LangChain4j Integration
// DEPS tech.kayys.gollek:gollek-sdk-langchain4j:1.0.0-SNAPSHOT
```

### Common Dependencies

```java
// SLF4J Logging
// DEPS org.slf4j:slf4j-simple:2.0.0

// JUnit for Testing
// DEPS junit:junit:4.13.2
// DEPS org.hamcrest:hamcrest-core:2.2

// JSON Processing
// DEPS com.google.code.gson:gson:2.10.1

// Apache Commons Lang
// DEPS org.apache.commons:commons-lang3:3.12.0
```

---

## Writing Your First jbang Script

### File Naming Convention

- **File name**: Use underscores: `my_script.java`
- **Class name**: Must match file name: `public class my_script`

❌ **Wrong:**
```java
// File: my-script.java
public class MyScript {  // ❌ Doesn't match filename
```

✅ **Correct:**
```java
// File: my_script.java
public class my_script {  // ✅ Matches filename
```

### Script Structure

```java
///usr/bin/env jbang
// DEPS <groupId>:<artifactId>:<version>

import java.util.*;
import tech.kayys.gollek.ml.nn.*;

/**
 * Description of what this script does
 */
public class my_script {
    
    public static void main(String[] args) {
        // Your code here
        System.out.println("Hello from jbang!");
    }
}
```

### Command-line Arguments

```java
public class process_data {
    public static void main(String[] args) {
        // Check arguments
        if (args.length < 2) {
            System.err.println("Usage: jbang process_data.java <input> <output>");
            System.exit(1);
        }
        
        String input = args[0];
        String output = args[1];
        
        // Process...
    }
}
```

**Run with arguments:**
```bash
jbang process_data.java input.txt output.txt
```

---

## Building Models

### Simple Linear Model

```java
///usr/bin/env jbang
// DEPS tech.kayys.gollek:gollek-sdk-nn:1.0.0-SNAPSHOT

import tech.kayys.gollek.ml.nn.*;

public class simple_model {
    public static void main(String[] args) {
        // Create a simple model
        Module model = new Sequential(
            new Linear(10, 5),
            new ReLU(),
            new Linear(5, 1)
        );
        
        System.out.println("✅ Model created successfully!");
        System.out.println("Architecture: 10 → 5 → 1");
    }
}
```

### Complex Model

```java
///usr/bin/env jbang
// DEPS tech.kayys.gollek:gollek-sdk-nn:1.0.0-SNAPSHOT

import tech.kayys.gollek.ml.nn.*;
import java.util.*;

public class complex_model {
    public static void main(String[] args) {
        // Build a more complex model
        Module model = new Sequential(
            new Linear(784, 256),
            new ReLU(),
            new Linear(256, 128),
            new ReLU(),
            new Linear(128, 64),
            new ReLU(),
            new Linear(64, 10)
        );
        
        System.out.println("✅ Deep model created!");
        System.out.println("Layers: 784 → 256 → 128 → 64 → 10");
        
        // Try different activations
        ReLU relu = new ReLU();
        Sigmoid sigmoid = new Sigmoid();
        Tanh tanh = new Tanh();
        
        System.out.println("✅ Activations available!");
    }
}
```

---

## Training Models

### Basic Training Loop

```java
///usr/bin/env jbang
// DEPS tech.kayys.gollek:gollek-sdk-nn:1.0.0-SNAPSHOT

import tech.kayys.gollek.ml.nn.*;

public class train_model {
    public static void main(String[] args) {
        // Create model
        Module model = new Sequential(
            new Linear(784, 128),
            new ReLU(),
            new Linear(128, 10)
        );
        
        // Loss and optimizer
        CrossEntropyLoss loss = new CrossEntropyLoss();
        Adam optimizer = new Adam(0.001);
        
        // Training loop
        int epochs = 10;
        for (int epoch = 0; epoch < epochs; epoch++) {
            // Forward pass
            // loss computation
            // backward pass
            // optimizer step
            
            System.out.println("Epoch " + (epoch + 1) + "/" + epochs);
        }
        
        System.out.println("✅ Training complete!");
    }
}
```

---

## Exporting and Sharing

### Export as JAR

```bash
# Build executable JAR
jbang -o my_app.jar my_script.java

# Run the JAR
java -jar my_app.jar arg1 arg2
```

### Run from GitHub

```bash
# Run script directly from GitHub
jbang https://raw.githubusercontent.com/user/repo/main/my_script.java
```

### Package with Maven

To include in your project:

1. Save script in `src/main/scripts/`
2. Configure `maven-assembly-plugin` to include scripts
3. Distribute as part of your release

---

## Troubleshooting

### "Class X is public, should be declared in a file named..."

**Cause:** Filename doesn't match class name.

**Fix:**
```bash
# If file is: my-script.java
# And class is: public class MyScript

# Rename file to match class name
mv my-script.java MyScript.java
# Change class to match file
# public class my_script { ... }
mv my_script.java my_script.java
```

### "Package X does not exist"

**Cause:** Dependency not found in Maven repositories.

**Fix 1:** Check DEPS syntax
```java
// ✅ Correct
// DEPS tech.kayys.gollek:gollek-sdk-nn:1.0.0-SNAPSHOT

// ❌ Wrong
// DEPS tech.kayys.gollek-sdk-nn:1.0.0
```

**Fix 2:** Build SDK locally
```bash
cd gollek/sdk
mvn clean install -DskipTests
```

**Fix 3:** Clear jbang cache
```bash
rm -rf ~/.jbang
# Retry
jbang my_script.java
```

### "Cannot find symbol: class X"

**Cause:** Missing import or dependency not loaded.

**Fix:**
```bash
# Run with verbose output
jbang --verbose my_script.java

# Check if dependency was downloaded
jbang --verbose -x my_script.java
```

---

## Performance Tips

### First Run (Slow - 10-30 seconds)
- jbang downloads dependencies
- Compiles Java source
- Caches compiled JAR

### Subsequent Runs (Fast - 1-2 seconds)
- Uses cached compiled JAR
- Skips download and compilation

**To speed up repeated runs:**
```bash
# Build as standalone JAR once
jbang -o my_app.jar my_script.java

# Then just run the JAR (instant)
java -jar my_app.jar
```

### Memory Management

Increase heap size if needed:

```bash
# Edit my_script.java
// JAVA_OPTS -Xmx2g -Xms1g

# Or set environment variable
export JAVA_OPTS="-Xmx2g -Xms1g"
jbang my_script.java
```

---

## Advanced Features

### Using IDE with jbang

Generate IDE project files:

```bash
# Generate Maven pom.xml
jbang -o pom.xml my_script.java

# Now open in IDE with Maven support
```

### Debugging

Run with debug server:

```bash
jbang --debug my_script.java
```

Then attach your IDE debugger to localhost:4711.

### Custom Repositories

Add additional Maven repositories:

```java
///usr/bin/env jbang
// REPOS jcenter
// REPOS https://my.company.com/maven

// DEPS my.company:my-lib:1.0.0
```

---

## Examples in gollek/sdk

### Working Examples

1. **gollek_template.java**
   - Basic jbang template
   - Demonstrates arguments, collections, strings
   - ✅ Works immediately

2. **examples/neural-network-example.java**
   - Neural network model creation
   - Training loop structure
   - Requires Gollek SDK (see installation above)

3. **examples/** directory
   - More complex patterns
   - Integration examples
   - Best practices

### Running Examples

```bash
# Basic template
jbang gollek/sdk/jbang-templates/gollek_template.java

# With arguments
jbang gollek/sdk/jbang-templates/gollek_template.java hello world

# Neural network (requires SDK built locally)
jbang gollek/sdk/jbang-templates/examples/neural-network-example.java
```

---

## Next Steps

1. **Run the template**: Get familiar with jbang syntax
2. **Modify the template**: Add your own logic
3. **Build SDK locally**: `mvn clean install`
4. **Add Gollek imports**: Use classes from gollek-sdk-nn
5. **Create custom scripts**: Build your own jbang scripts
6. **Publish to Maven Central**: Share with others (optional)

---

## Resources

- **jbang Official**: https://jbang.dev/
- **jbang Documentation**: https://jbang.dev/documentation/
- **Gollek SDK**: See JBANG_SETUP.md and examples/
- **Maven Central**: https://mvnrepository.com/

---

## Support

For issues with jbang:
- Check jbang troubleshooting: `jbang --help`
- Clear cache: `rm -rf ~/.jbang`
- Run with verbose: `jbang --verbose my_script.java`

For issues with Gollek SDK:
- Check local build: `mvn clean install`
- See TROUBLESHOOTING.md in gollek/sdk/
- Review examples/ directory for patterns

---

**Happy scripting! 🚀**

For Jupyter integration, see: JUPYTER_SETUP.md
For combined workflows, see: EXAMPLES.md
