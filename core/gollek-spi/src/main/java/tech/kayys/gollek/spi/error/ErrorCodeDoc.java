package tech.kayys.gollek.spi.error;

import java.util.ArrayList;
import java.util.List;

/**
 * Utility to generate Markdown documentation for {@link ErrorCode}.
 */
public final class ErrorCodeDoc {

    private ErrorCodeDoc() {
    }

    public static String toMarkdown() {
        String nl = System.lineSeparator();
        StringBuilder builder = new StringBuilder();
        builder.append("# Gollek Error Codes").append(nl).append(nl);
        builder.append("Generated from `ErrorCode` at build time.").append(nl).append(nl);

        for (ErrorCode.ErrorCategory category : ErrorCode.ErrorCategory.values()) {
            List<ErrorCode> codes = byCategoryInOrder(category);
            if (codes.isEmpty()) {
                continue;
            }
            builder.append("## ").append(category.getPrefix()).append(nl).append(nl);
            builder.append("| Code | HTTP | Retryable | Message |").append(nl);
            builder.append("| --- | --- | --- | --- |").append(nl);
            for (ErrorCode code : codes) {
                builder.append("| ")
                        .append(code.getCode()).append(" | ")
                        .append(code.getHttpStatus()).append(" | ")
                        .append(code.isRetryable()).append(" | ")
                        .append(code.getDefaultMessage())
                        .append(" |").append(nl);
            }
            builder.append(nl);
        }

        return builder.toString();
    }

    public static void main(String[] args) {
        System.out.print(toMarkdown());
    }

    private static List<ErrorCode> byCategoryInOrder(ErrorCode.ErrorCategory category) {
        List<ErrorCode> codes = new ArrayList<>();
        for (ErrorCode errorCode : ErrorCode.values()) {
            if (errorCode.getCategory() == category) {
                codes.add(errorCode);
            }
        }
        return codes;
    }
}
