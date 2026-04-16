package tech.kayys.gollek.jupyter;

import jdk.jshell.JShell;
import org.dflib.jjava.jupyter.kernel.ReplacementOptions;

import java.util.List;
import java.util.stream.Collectors;

/**
 * Code completion backed by JShell's built-in completion engine,
 * augmented with Gollek SDK method hints.
 */
public class CompletionProvider {

    private final JShell shell;

    public CompletionProvider(JShell shell) {
        this.shell = shell;
    }

    public ReplacementOptions complete(String code, int cursor) {
        int[] anchor = new int[1];
        List<String> suggestions = shell.sourceCodeAnalysis()
                .completionSuggestions(code, cursor, anchor)
                .stream()
                .map(s -> s.continuation())
                .distinct()
                .sorted()
                .limit(100)
                .collect(Collectors.toList());

        return suggestions.isEmpty() ? null
                : new ReplacementOptions(suggestions, anchor[0], cursor);
    }
}
