# Troubleshooting Guide - Jupyter & jbang

Solutions to common problems when using Gollek SDK with Jupyter and jbang.

## Jupyter Issues

### Installation Problems

#### Issue: `jupyter command not found`

**Cause**: Jupyter not installed or not in PATH

**Solutions**:
```bash
# Install Jupyter
pip install jupyter

# Or with conda
conda install jupyter

# Verify installation
jupyter --version
```

#### Issue: `java.lang.ClassNotFoundException`

**Cause**: Gollek SDK JAR not in classpath

**Solutions**:
```bash
# Rebuild Gollek SDK
cd gollek/sdk
mvn clean install -DskipTests

# Reinstall kernel
bash jupyter-kernel/install.sh

# Restart Jupyter
```

#### Issue: IJava kernel not found

**Cause**: Java kernel not installed

**Solutions**:
```bash
# Install IJava kernel
pip install ijava

# Or build from source
git clone https://github.com/SpencerPark/IJava.git
cd IJava
mvn clean install
python install.py --sys-prefix

# Verify
jupyter kernelspec list
```

### Runtime Issues

#### Issue: `OutOfMemoryError`

**Cause**: Java heap too small for large models

**Solutions**:
```bash
# Method 1: Edit kernel.json
# Add to kernel.json argv:
"java", "-Xmx4G", ...

# Method 2: Environment variable
export _JAVA_OPTIONS="-Xmx4G"
jupyter notebook

# Method 3: In notebook cell
System.setProperty("java.util.concurrent.ForkJoinPool.common.parallelism", "4");
```

#### Issue: Very slow first execution

**Cause**: JIT compilation on first run

**Solutions**:
- This is normal; be patient (30-60 seconds)
- Subsequent cells run much faster
- Pre-warm by importing everything first:

```java
import tech.kayys.gollek.ml.nn.*;
import tech.kayys.gollek.ml.autograd.*;
// Runs slow, but subsequent code is fast
```

#### Issue: Import errors (symbol not found)

**Cause**: Missing dependencies or wrong package name

**Solutions**:
```bash
# Check what's actually installed
mvn dependency:tree -DoutputFile=deps.txt
cat deps.txt

# Update pom.xml with missing dependencies
# Rebuild Gollek SDK
mvn clean install -DskipTests

# Reinstall kernel
bash jupyter-kernel/install.sh
```

#### Issue: Connection timeout

**Cause**: Kernel crashed or hung

**Solutions**:
1. Interrupt kernel (Kernel → Interrupt in menu)
2. Restart kernel (Kernel → Restart)
3. Restart Jupyter entirely
4. Check system logs for Java crashes

### Usage Issues

#### Issue: Variables not persisting between cells

**This is normal behavior**. Variables ARE persisted. Check:
- Are you in the same notebook?
- Did you restart the kernel?
- Is the cell actually executing? (Look for execution number)

#### Issue: Output not showing

**Solutions**:
- Make sure you print output:
```java
System.out.println("This shows");
// vs
"this doesn't show";  // Need explicit println
```

#### Issue: Large output truncating

**Solutions**:
```java
// Jupyter truncates very large outputs. Use:
String[] parts = new String[10000];  // Large array
// Instead, print summary:
System.out.println("Array length: " + parts.length);
System.out.println("First 5: " + Arrays.toString(Arrays.copyOf(parts, 5)));
```

## jbang Issues

### Installation Problems

#### Issue: `jbang command not found`

**Cause**: jbang not installed or not in PATH

**Solutions**:
```bash
# Install jbang
curl -Ls https://sh.jbang.dev | bash -s - app setup

# Add to PATH
export PATH=$PATH:~/.jbang/bin

# Verify
jbang --version
```

#### Issue: Permissions denied

**Cause**: Script not executable

**Solutions**:
```bash
# Make script executable
chmod +x myscript.java

# Or just use jbang to run it
jbang myscript.java
```

### Runtime Issues

#### Issue: `Dependency not found`

**Cause**: Invalid dependency format or artifact doesn't exist

**Solutions**:
```java
// Correct format: group:artifact:version
// DEPS tech.kayys:gollek-sdk-nn:1.0.0

// Check available versions
# https://mvnrepository.com

// Verify Maven repository
// REPOS https://repo.maven.apache.org/maven2
```

#### Issue: Class not found at runtime

**Cause**: Missing dependency or wrong class path

**Solutions**:
```bash
# Check resolved dependencies
jbang --info myscript.java

# Clear cache and retry
rm -rf ~/.jbang
jbang myscript.java

# Verify dependency is correct
// DEPS tech.kayys:gollek-sdk-nn:1.0.0
```

#### Issue: Out of memory

**Cause**: Heap too small for processing

**Solutions**:
```bash
# Method 1: Command line
jbang -Xmx2G myscript.java

# Method 2: In script
// RUNTIME_OPTIONS -Xmx2G

# Method 3: Environment
export JBANG_OPTS="-Xmx2G"
jbang myscript.java
```

#### Issue: Slow first run

**Cause**: Downloading dependencies and JIT compilation

**Solutions**:
```bash
# First run takes time (1-5 minutes)
# Be patient; subsequent runs are fast

# Or export as JAR for reuse
jbang -o myscript.jar myscript.java
java -jar myscript.jar  # Much faster
```

### Usage Issues

#### Issue: Script exits immediately

**Cause**: No output or exception thrown

**Solutions**:
```java
// Add error handling
try {
    // Your code
} catch (Exception e) {
    e.printStackTrace();
    System.exit(1);
}

// Add logging
System.out.println("Starting...");
// code
System.out.println("Done!");
```

#### Issue: Arguments not being read

**Cause**: Not accessing args array correctly

**Solutions**:
```java
// Correct way
public static void main(String[] args) {
    if (args.length > 0) {
        System.out.println("Got: " + args[0]);
    }
}

// Run with
jbang script.java arg1 arg2
```

#### Issue: Can't find local files

**Cause**: Working directory or path issue

**Solutions**:
```java
// Use absolute paths
Path file = Paths.get("/absolute/path/to/file");

// Or current directory
Path file = Paths.get("./relative/path");

// Check where you are
System.out.println("CWD: " + System.getProperty("user.dir"));
```

## Common Errors

### `NoClassDefFoundError`

```
Error: Could not find or load main class...
```

**Solution**:
- Check DEPS syntax
- Verify artifact exists
- Clear ~/.jbang cache

### `ClassCastException`

```
Cannot cast ... to ...
```

**Solution**:
- Check types match
- Verify no library version conflicts
- Review import statements

### `NullPointerException`

```
Exception in thread "main" java.lang.NullPointerException
```

**Solution**:
- Add null checks:
```java
if (obj != null) {
    // use obj
}
```

### `FileNotFoundException`

```
File not found: ...
```

**Solution**:
- Check file path is correct
- Verify file exists
- Use absolute paths

## Debugging Techniques

### Jupyter Debugging

```java
// Add verbose output
System.out.println("DEBUG: Starting...");
System.out.println("DEBUG: x = " + x);
System.out.println("DEBUG: Done!");

// Check types
System.out.println(value.getClass());

// Print arrays
System.out.println(Arrays.toString(array));

// Print object
System.out.println(obj);
```

### jbang Debugging

```bash
# Run with debug flag
jbang --debug myscript.java

# Run with stack traces
jbang myscript.java 2>&1 | tee output.log

# Add verbose output
jbang -verbose myscript.java
```

## Performance Optimization

### Jupyter

```java
// Batch operations
// ❌ Slow: process one at a time
// ✓ Fast: process in batches

// Reuse objects
Module model = new Sequential(...);  // Create once
// Use model many times

// Avoid re-imports
// Import at start, not in every cell
```

### jbang

```java
// Use shaded JAR
jbang -o fat.jar script.java

// Cache results
// ❌ Slow: re-compute every time
// ✓ Fast: save and load results

// Optimize dependencies
// ❌ Many: slows down startup
// ✓ Few: faster execution
```

## Getting Help

### Step 1: Identify the problem
- Read error message carefully
- Check this guide
- Search documentation

### Step 2: Collect information
- Error message
- Operating system
- Java version (`java -version`)
- Jupyter/jbang version

### Step 3: Try solutions
- Follow steps in this guide
- Check GitHub issues
- Review examples

### Step 4: Report issue
- Include error message
- Include reproduction steps
- Include environment info

## Quick Reference

| Problem | Solution |
|---------|----------|
| Import failed | Rebuild Gollek SDK, reinstall kernel |
| Out of memory | Increase heap: `-Xmx4G` |
| Slow first run | Normal; subsequent runs are fast |
| Dependency not found | Check version, verify repository |
| File not found | Use absolute path or check cwd |
| Class not found | Add DEPS, clear ~/.jbang cache |
| Kernel crash | Restart kernel, check logs |
| No output | Add System.out.println() statements |

## Resources

- [Jupyter Documentation](https://jupyter.org/documentation)
- [jbang Documentation](https://jbang.dev/)
- [Gollek SDK Support](../README.md)

---

**For further help, check the main documentation or open an issue!**
