package tech.kayys.gollek.cli.chat;

import jakarta.enterprise.context.Dependent;
import org.jline.reader.*;
import org.jline.terminal.Terminal;
import org.jline.terminal.TerminalBuilder;
import org.jline.widget.AutosuggestionWidgets;
import org.jline.widget.TailTipWidgets;
import org.jline.console.CmdDesc;

import java.nio.file.Path;
import java.util.Map;

/**
 * Manages JLine Terminal and LineReader for the chat session.
 */
@Dependent
public class ChatTerminalHandler implements AutoCloseable {

    private Terminal terminal;
    private LineReader reader;
    public void initialize(boolean quiet, Completer completer, Map<String, CmdDesc> commandHelp) {
        try {
            try {
                terminal = TerminalBuilder.builder()
                        .system(true)
                        .dumb(true)
                        .build();
            } catch (Exception e) {
                terminal = TerminalBuilder.builder().dumb(true).build();
            }

            reader = LineReaderBuilder.builder()
                    .terminal(terminal)
                    .completer(completer)
                    .variable(LineReader.HISTORY_FILE,
                            Path.of(System.getProperty("user.home"), ".gollek", "chat_history"))
                    .variable(LineReader.LIST_MAX, 50)
                    .option(LineReader.Option.AUTO_MENU, true)
                    .option(LineReader.Option.AUTO_LIST, true)
                    .option(LineReader.Option.COMPLETE_IN_WORD, true)
                    .build();

            // Enable autosuggestions
            AutosuggestionWidgets autosuggestionWidgets = new AutosuggestionWidgets(reader);
            autosuggestionWidgets.enable();

            // Enable TailTip
            if (commandHelp != null && !commandHelp.isEmpty()) {
                TailTipWidgets tailTipWidgets = new TailTipWidgets(reader, commandHelp, 0,
                        TailTipWidgets.TipType.COMPLETER);
                tailTipWidgets.enable();
            }

        } catch (Exception e) {
            throw new RuntimeException("Failed to initialize terminal: " + e.getMessage(), e);
        }
    }

    public String readInput(String prompt, String secondaryPrompt) {
        StringBuilder inputBuffer = new StringBuilder();
        String currentPrompt = prompt;

        while (true) {
            String lineInput;
            try {
                lineInput = reader.readLine(currentPrompt);
            } catch (UserInterruptException e) {
                return null; // Interrupted
            } catch (EndOfFileException e) {
                throw e; // Propagate EOF
            }

            if (lineInput.endsWith("\\")) {
                inputBuffer.append(lineInput, 0, lineInput.length() - 1).append("\n");
                currentPrompt = secondaryPrompt;
            } else {
                inputBuffer.append(lineInput);
                break;
            }
        }
        return inputBuffer.toString().trim();
    }

    public Terminal getTerminal() {
        return terminal;
    }

    public LineReader getReader() {
        return reader;
    }

    @Override
    public void close() throws Exception {
        if (terminal != null) {
            terminal.close();
        }
    }
}
