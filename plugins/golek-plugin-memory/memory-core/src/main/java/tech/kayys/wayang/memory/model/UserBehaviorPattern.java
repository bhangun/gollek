package tech.kayys.wayang.memory.model;

public class UserBehaviorPattern {
    private final String pattern;
    private final String description;
    private final double strength;

    public UserBehaviorPattern(
            String pattern,
            String description,
            double strength) {
        this.pattern = pattern;
        this.description = description;
        this.strength = strength;
    }

    public String getPattern() { return pattern; }
    public String getDescription() { return description; }
    public double getStrength() { return strength; }
}