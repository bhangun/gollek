# jbang Templates - Gollek SDK Integration

This directory contains jbang templates for using Gollek SDK in single-file Java scripts.

## Quick Start

### 1. Run the Basic Template

```bash
cd /path/to/gollek/sdk
jbang jbang-templates/gollek_template.java
```

**Expected output:**
```
🚀 Gollek SDK jbang Template
============================

Example 1: Processing Arguments
-------------------------------
No arguments provided.
Try: jbang gollek_template.java hello world

[more examples...]

✅ Template examples completed successfully!
```

### 2. Try with Arguments

```bash
jbang jbang-templates/gollek_template.java arg1 arg2 arg3
```

### 3. Use as Template for Your Own Script

```bash
# Copy template
cp jbang-templates/gollek_template.java my_script.java

# Edit my_script.java
# - Change class name: `public class my_script`
# - Add your logic

# Run your script
jbang my_script.java
```

---

## Files

### gollek_template.java
- **Status**: ✅ Working
- **Purpose**: Basic jbang template demonstrating syntax
- **Features**: 
  - Command-line argument processing
  - Data processing (collections, streams)
  - String manipulation
  - No external dependencies (standalone)
- **Run**: `jbang gollek_template.java [args]`

### examples/neural-network-example.java
- **Status**: Requires Gollek SDK local build
- **Purpose**: Shows how to use Gollek SDK classes
- **Features**:
  - Neural network model creation
  - Layer composition
  - Training structure
- **Setup Required**: `mvn clean install` in gollek/sdk/
- **Run**: `jbang examples/neural-network-example.java`

---

## Important: File Naming Convention

### Java Filename vs Class Name

jbang requires that **public class names match the filename** (with `.java` extension).

#### ✅ Correct

**File: `my_script.java`**
```java
public class my_script {
    public static void main(String[] args) {
        // code
    }
}
```

**File: `goldlek_template.java`**
```java
public class gollek_template {
    public static void main(String[] args) {
        // code
    }
}
```

#### ❌ Wrong

**File: `my-script.java`** (with hyphen)
```java
public class MyScript {  // ❌ Doesn't match filename
    // code
}
```

**File: `my_script.java`**
```java
public class MyScript {  // ❌ Class name doesn't match filename
    // code
}
```

---

## Using Gollek SDK Classes

### Step 1: Build SDK Locally

```bash
cd /path/to/gollek/sdk
mvn clean install -DskipTests
```

This publishes Gollek SDK to your local Maven repository (`~/.m2/repository`).

### Step 2: Reference in DEPS

Add this line at the top of your jbang script:

```java
///usr/bin/env jbang
// DEPS tech.kayys.gollek:gollek-sdk-nn:1.0.0-SNAPSHOT
// DEPS org.slf4j:slf4j-simple:2.0.0

import tech.kayys.gollek.ml.nn.*;

public class my_model {
    public static void main(String[] args) {
        // Use Gollek classes here
    }
}
```

### Step 3: Run

```bash
jbang my_model.java
```

First run: 10-30 seconds (downloads dependencies, compiles)
Subsequent runs: 1-2 seconds (uses cached JAR)

---

## Common Patterns

### Pattern 1: Simple Model Creation

```java
///usr/bin/env jbang
// DEPS tech.kayys.gollek:gollek-sdk-nn:1.0.0-SNAPSHOT

import tech.kayys.gollek.ml.nn.*;

public class create_model {
    public static void main(String[] args) {
        Module model = new Sequential(
            new Linear(10, 5),
            new ReLU(),
            new Linear(5, 1)
        );
        System.out.println("✅ Model: " + model);
    }
}
```

### Pattern 2: CLI Tool with Arguments

```java
///usr/bin/env jbang
// DEPS tech.kayys.gollek:gollek-sdk-nn:1.0.0-SNAPSHOT

import tech.kayys.gollek.ml.nn.*;

public class train_cli {
    public static void main(String[] args) {
        if (args.length < 2) {
            System.err.println("Usage: jbang train_cli epochs batchSize");
            System.exit(1);
        }
        
        int epochs = Integer.parseInt(args[0]);
        int batchSize = Integer.parseInt(args[1]);
        
        // Build and train model
    }
}
```

**Usage:**
```bash
jbang train_cli.java 10 32
```

### Pattern 3: Data Processing

```java
///usr/bin/env jbang
// DEPS tech.kayys.gollek:gollek-sdk-data:1.0.0-SNAPSHOT

import tech.kayys.gollek.data.*;
import java.util.*;

public class process_data {
    public static void main(String[] args) {
        // Load data
        // Process with Gollek
        // Save results
    }
}
```

---

## Troubleshooting

### Error: "Class X is public, should be declared in a file named..."

**Cause**: Filename doesn't match public class name.

**Fix**: Ensure filename (without `.java`) matches class name exactly.

```bash
# If you have: public class GolleKTemplate { }
# File must be named: GolleKTemplate.java

# If you have: public class gollek_template { }
# File must be named: gollek_template.java
```

### Error: "Package X does not exist"

**Cause**: Gollek SDK not built locally, or wrong DEPS syntax.

**Solutions**:

1. Build SDK locally:
   ```bash
   cd gollek/sdk && mvn clean install -DskipTests
   ```

2. Check DEPS syntax (use colons, not dashes):
   ```java
   // ✅ Correct
   // DEPS tech.kayys.gollek:gollek-sdk-nn:1.0.0-SNAPSHOT
   
   // ❌ Wrong
   // DEPS tech.kayys.gollek-sdk-nn:1.0.0
   ```

3. Clear cache if updated:
   ```bash
   rm -rf ~/.jbang
   ```

### Error: "Cannot find symbol: class Linear"

**Cause**: Import statement missing or SDK not in DEPS.

**Fix**: Add both DEPS and imports:

```java
///usr/bin/env jbang
// DEPS tech.kayys.gollek:gollek-sdk-nn:1.0.0-SNAPSHOT

import tech.kayys.gollek.ml.nn.*;  // ← Add this import
```

### Slow First Run (10-30 seconds)

**Normal behavior**:
- jbang downloads Maven dependencies
- Java source is compiled to JAR
- Results are cached for next runs

**Subsequent runs are fast** (1-2 seconds).

To speed up by pre-building:
```bash
jbang -o my_app.jar my_script.java
java -jar my_app.jar  # Fast execution
```

---

## Advanced Features

### Export as Standalone JAR

```bash
# Build as single JAR file
jbang -o my_app.jar my_script.java

# Share or run the JAR
java -jar my_app.jar arg1 arg2
```

### Run from GitHub

```bash
# Execute script directly from GitHub
jbang https://raw.githubusercontent.com/user/repo/main/my_script.java
```

### Custom Java Options

```java
///usr/bin/env jbang
// JAVA_OPTS -Xmx2g -Xms1g
// DEPS tech.kayys.gollek:gollek-sdk-nn:1.0.0-SNAPSHOT
```

### Use IDE with jbang

```bash
# Generate Maven pom.xml for IDE
jbang -o pom.xml my_script.java

# Now import into your favorite IDE
```

---

## Next Steps

1. **Start with gollek_template.java**: Get familiar with jbang
2. **Modify the template**: Add your own logic
3. **Build SDK**: `mvn clean install -DskipTests`
4. **Use Gollek classes**: Add DEPS and imports
5. **Create custom scripts**: Build your applications
6. **Share scripts**: Export as JARs or GitHub gists

---

## Resources

- **This directory**: See examples/ for more patterns
- **jbang Official**: https://jbang.dev/
- **Gollek SDK**: See other .md files in gollek/sdk/
- **Maven Central**: https://mvnrepository.com/

---

## Documentation Files

- **JBANG_SETUP_GUIDE.md**: Comprehensive setup guide
- **../EXAMPLES.md**: More example patterns
- **../TROUBLESHOOTING.md**: Common issues
- **../JUPYTER_JBANG_README.md**: Overview

---

**Happy scripting! 🚀**
