package tech.kayys.wayang.memory.context;

import java.util.List;

public class TaskPattern {
    private final String taskType;
    private final int frequency;
    private final List<String> commonSteps;
    private final double successRate;

    public TaskPattern(String taskType, int frequency, List<String> commonSteps, double successRate) {
        this.taskType = taskType;
        this.frequency = frequency;
        this.commonSteps = commonSteps;
        this.successRate = successRate;
    }

    public String getTaskType() { return taskType; }
    public int getFrequency() { return frequency; }
    public List<String> getCommonSteps() { return commonSteps; }
    public double getSuccessRate() { return successRate; }
}
