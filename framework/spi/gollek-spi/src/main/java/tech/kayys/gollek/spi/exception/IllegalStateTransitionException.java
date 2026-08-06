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
 */

package tech.kayys.gollek.spi.exception;

/**
 * Exception thrown when an illegal state transition is attempted.
 *
 * Used by state machines to enforce valid state transitions.
 */
public class IllegalStateTransitionException extends RuntimeException {

    private static final long serialVersionUID = 1L;

    /**
     * Constructs with a detail message.
     */
    public IllegalStateTransitionException(String message) {
        super(message);
    }

    /**
     * Constructs with a detail message and cause.
     */
    public IllegalStateTransitionException(String message, Throwable cause) {
        super(message, cause);
    }

    /**
     * Constructs with from-state and to-state.
     */
    public IllegalStateTransitionException(String fromState, String toState) {
        super("Illegal state transition: " + fromState + " -> " + toState);
    }
}
