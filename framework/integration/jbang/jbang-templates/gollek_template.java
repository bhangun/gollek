///usr/bin/env jbang
// DEPS org.slf4j:slf4j-simple:2.0.0

import java.util.*;

/**
 * Gollek SDK - jbang Template (Minimal Working Example)
 * 
 * This template demonstrates jbang syntax and structure.
 * For full Gollek SDK integration, see JBANG_SETUP.md for:
 * - Using local Maven artifacts
 * - Publishing to Maven Central
 * - Creating custom dependencies
 * 
 * Usage:
 *   jbang gollek_template.java
 *   jbang gollek_template.java arg1 arg2
 */
public class gollek_template {

    public static void main(String[] args) {
        System.out.println("🚀 Gollek SDK jbang Template");
        System.out.println("============================");
        System.out.println();
        
        // Example 1: Command-line arguments
        System.out.println("Example 1: Processing Arguments");
        System.out.println("-------------------------------");
        if (args.length > 0) {
            System.out.println("Received " + args.length + " arguments:");
            for (int i = 0; i < args.length; i++) {
                System.out.println("  [" + i + "] " + args[i]);
            }
        } else {
            System.out.println("No arguments provided.");
            System.out.println("Try: jbang gollek_template.java hello world");
        }
        System.out.println();
        
        // Example 2: Collections
        System.out.println("Example 2: Data Processing");
        System.out.println("--------------------------");
        List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5);
        int sum = numbers.stream().mapToInt(Integer::intValue).sum();
        System.out.println("Sum of " + numbers + " = " + sum);
        System.out.println();
        
        // Example 3: String manipulation
        System.out.println("Example 3: String Operations");
        System.out.println("---------------------------");
        String text = "Gollek SDK jbang Integration";
        System.out.println("Original: " + text);
        System.out.println("Uppercase: " + text.toUpperCase());
        System.out.println("Reversed: " + new StringBuilder(text).reverse());
        System.out.println();
        
        System.out.println("✅ Template examples completed successfully!");
        System.out.println();
        System.out.println("📚 To use Gollek SDK in your scripts:");
        System.out.println("  1. See: jbang-templates/JBANG_SETUP.md");
        System.out.println("  2. See: examples/neural-network-example.java");
        System.out.println("  3. Build SDK locally: mvn clean install");
        System.out.println("  4. Reference in DEPS: // DEPS <groupId>:<artifactId>:<version>");
    }
}
