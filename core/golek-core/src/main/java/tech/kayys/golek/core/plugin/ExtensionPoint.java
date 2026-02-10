/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
 *
 * @author Bhangun
 */
package tech.kayys.golek.core.plugin;

/**
 * Extension point interface
 */
public interface ExtensionPoint {
    /**
     * Get extension point ID
     */
    String getId();

    /**
     * Get extension point name
     */
    String getName();

    /**
     * Get extension point type
     */
    Class<?> getExtensionType();

    /**
     * Get extension point description
     */
    String getDescription();
}