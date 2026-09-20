package main

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
)

type execResult struct {
	ExitCode        int    `json:"exit_code"`
	Stdout          string `json:"stdout"`
	Stderr          string `json:"stderr"`
	TimedOut        bool   `json:"timed_out"`
	StdoutTruncated bool   `json:"stdout_truncated"`
	StderrTruncated bool   `json:"stderr_truncated"`
}

type executionError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

type invocationState struct {
	Version      int             `json:"version"`
	InvocationID string          `json:"invocation_id"`
	Status       string          `json:"status"`
	Result       *execResult     `json:"result,omitempty"`
	Error        *executionError `json:"error,omitempty"`
}

func state(id, status string) invocationState {
	return invocationState{Version: 1, InvocationID: id, Status: status}
}

// invocationStore is compute-scoped by default. One manager owns a root;
// its mutex serializes start/get with registration and terminal publication.
type invocationStore struct {
	root string
}

func (s invocationStore) dir(id string) string {
	return filepath.Join(s.root, id)
}

func (s invocationStore) read(id string) (invocationState, error) {
	data, err := os.ReadFile(filepath.Join(s.dir(id), "result.json"))
	if err == nil {
		var result invocationState
		err = json.Unmarshal(data, &result)
		return result, err
	}
	if !errors.Is(err, os.ErrNotExist) {
		return invocationState{}, err
	}
	if _, err = os.Stat(filepath.Join(s.dir(id), "started.json")); err == nil {
		return state(id, "interrupted"), nil // The manager checks for a live owner.
	}
	if errors.Is(err, os.ErrNotExist) {
		return state(id, "not_found"), nil
	}
	return invocationState{}, err
}

func (s invocationStore) start(id string) error {
	if err := os.Mkdir(s.dir(id), 0700); err != nil {
		return err
	}
	// Do not persist commands or environment variables: they may contain secrets.
	return writeRecord(filepath.Join(s.dir(id), "started.json"), state(id, "in_progress"))
}

func (s invocationStore) finish(result invocationState) error {
	return writeRecord(filepath.Join(s.dir(result.InvocationID), "result.json"), result)
}

func writeRecord(path string, value invocationState) error {
	data, err := json.Marshal(value)
	if err != nil {
		return err
	}
	f, err := os.CreateTemp(filepath.Dir(path), ".record-*")
	if err != nil {
		return err
	}
	defer os.Remove(f.Name())
	if _, err = f.Write(data); err == nil {
		err = f.Sync()
	}
	closeErr := f.Close()
	if err != nil {
		return err
	}
	if closeErr != nil {
		return closeErr
	}
	return os.Rename(f.Name(), path)
}
