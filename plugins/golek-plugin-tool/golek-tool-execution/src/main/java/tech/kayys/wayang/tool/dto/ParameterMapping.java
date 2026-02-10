package tech.kayys.wayang.tool.dto;

public record ParameterMapping(
    String name,
    String location, // path, query, header
    Boolean required,
    String description
) { }
