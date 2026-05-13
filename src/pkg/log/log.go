// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

// Package log provides structured logging for Slemify.
// In text mode (default), outputs human-friendly messages.
// In JSON mode, outputs structured JSON lines for machine consumption.
package log

import (
	"io"
	"log/slog"
	"os"
)

var logger = slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelInfo}))

// Init configures the global logger based on output format.
// Call this once at startup after parsing flags.
func Init(format string) {
	var handler slog.Handler
	switch format {
	case "json":
		handler = slog.NewJSONHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelInfo})
	default:
		handler = slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelInfo})
	}
	logger = slog.New(handler)
	slog.SetDefault(logger)
}

// Logger returns the configured slog.Logger for direct use.
func Logger() *slog.Logger {
	return logger
}

// Writer returns an io.Writer that writes to the logger at Info level.
// Useful for passing to libraries that accept an io.Writer.
func Writer() io.Writer {
	return os.Stderr
}

// Info logs at Info level.
func Info(msg string, args ...any) {
	logger.Info(msg, args...)
}

// Warn logs at Warn level.
func Warn(msg string, args ...any) {
	logger.Warn(msg, args...)
}

// Error logs at Error level.
func Error(msg string, args ...any) {
	logger.Error(msg, args...)
}

// Debug logs at Debug level.
func Debug(msg string, args ...any) {
	logger.Debug(msg, args...)
}
