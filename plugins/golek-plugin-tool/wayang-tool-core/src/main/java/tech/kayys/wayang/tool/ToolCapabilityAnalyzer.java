package tech.kayys.wayang.tool;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.wayang.tool.dto.CapabilityLevel;

/**
 * Tool capability analyzer
 */
@ApplicationScoped
public class ToolCapabilityAnalyzer {

    public CapabilityLevel analyze(
            io.swagger.v3.oas.models.PathItem.HttpMethod method,
            io.swagger.v3.oas.models.Operation operation,
            String path) {

        // Determine capability level based on HTTP method and path
        if (method == io.swagger.v3.oas.models.PathItem.HttpMethod.GET) {
            return CapabilityLevel.READ_ONLY;
        } else if (method == io.swagger.v3.oas.models.PathItem.HttpMethod.POST ||
                method == io.swagger.v3.oas.models.PathItem.HttpMethod.PUT ||
                method == io.swagger.v3.oas.models.PathItem.HttpMethod.PATCH) {

            // Check if operation affects system state
            if (isStateChangingOperation(operation, path)) {
                return CapabilityLevel.STATE_CHANGING;
            } else {
                return CapabilityLevel.DATA_MANIPULATION;
            }
        } else if (method == io.swagger.v3.oas.models.PathItem.HttpMethod.DELETE) {
            return CapabilityLevel.DESTRUCTIVE;
        }

        return CapabilityLevel.UNKNOWN;
    }

    private boolean isStateChangingOperation(
            io.swagger.v3.oas.models.Operation operation,
            String path) {

        // Operations that typically change system state
        // This is a simplified heuristic - in practice, you'd need more sophisticated
        // analysis
        String operationId = operation.getOperationId();
        if (operationId != null) {
            operationId = operationId.toLowerCase();

            // Common destructive/state-changing verbs
            return operationId.contains("create") ||
                    operationId.contains("update") ||
                    operationId.contains("delete") ||
                    operationId.contains("modify") ||
                    operationId.contains("change") ||
                    operationId.contains("set") ||
                    operationId.contains("configure") ||
                    operationId.contains("activate") ||
                    operationId.contains("deactivate") ||
                    operationId.contains("enable") ||
                    operationId.contains("disable") ||
                    operationId.contains("install") ||
                    operationId.contains("uninstall") ||
                    operationId.contains("deploy") ||
                    operationId.contains("undeploy") ||
                    operationId.contains("start") ||
                    operationId.contains("stop") ||
                    operationId.contains("restart") ||
                    operationId.contains("shutdown") ||
                    operationId.contains("terminate") ||
                    operationId.contains("suspend") ||
                    operationId.contains("resume");
        }

        // Fallback: check path for state-changing patterns
        String lowerPath = path.toLowerCase();
        return lowerPath.contains("settings") ||
                lowerPath.contains("config") ||
                lowerPath.contains("preferences") ||
                lowerPath.contains("profile") ||
                lowerPath.contains("account") ||
                lowerPath.contains("user") ||
                lowerPath.contains("admin") ||
                lowerPath.contains("system") ||
                lowerPath.contains("control");
    }
}