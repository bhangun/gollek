package tech.kayys.gollek.jupyter;

import org.dflib.jjava.execution.CodeEvaluator;
import org.dflib.jjava.execution.CodeEvaluatorBuilder;
import org.dflib.jjava.jupyter.kernel.BaseKernel;
import org.dflib.jjava.jupyter.kernel.LanguageInfo;
import org.dflib.jjava.jupyter.kernel.ReplacementOptions;
import org.dflib.jjava.jupyter.kernel.display.DisplayData;

import java.util.List;

public class GollekKernel extends BaseKernel {

    private static final LanguageInfo LANGUAGE_INFO = new LanguageInfo.Builder("java")
            .version(Runtime.version().toString())
            .mimetype("text/x-java-source")
            .fileExtension(".jshell")
            .pygments("java")
            .codemirror("java")
            .build();

    private final CodeEvaluator evaluator;
    private final CompletionProvider completer;

    public GollekKernel() {
        this.evaluator = new CodeEvaluatorBuilder()
                .sysStdout().sysStderr().sysStdin()
                .compilerOpts("--enable-preview")
                .build();
        this.completer = new CompletionProvider(evaluator.getShell());
    }

    @Override
    public DisplayData eval(String code) throws Exception {
        Object result = evaluator.eval(code);
        if (result == null) return DisplayData.EMPTY;
        DisplayData rich = TensorDisplay.render(result, getRenderer());
        return rich != null ? rich : new DisplayData(result.toString());
    }

    @Override
    public ReplacementOptions complete(String code, int cursor) throws Exception {
        return completer.complete(code, cursor);
    }

    @Override
    public String isComplete(String code) { return evaluator.isComplete(code); }

    @Override
    public void interrupt() { evaluator.interrupt(); }

    @Override
    public void onShutdown(boolean restart) { evaluator.shutdown(); }

    @Override
    public LanguageInfo getLanguageInfo() { return LANGUAGE_INFO; }

    @Override
    public String getBanner() {
        return "Gollek Kernel — Java " + Runtime.version() + "\nGollek ML libs on classpath";
    }

    @Override
    public List<LanguageInfo.Help> getHelpLinks() {
        return List.of(new LanguageInfo.Help("Gollek Docs", "https://gollek-ai.github.io/docs/"));
    }
}
