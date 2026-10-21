package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"time"
)

const outputLimit = 256 * 1024 // Per stream; continue draining after this limit.
const outputGrace = 100 * time.Millisecond

var invocationIDPattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$`)

type invocationRequest struct {
	Action  string            `json:"action,omitempty"`
	Runtime *invocationConfig `json:"_agentcore_runtime,omitempty"`
	Command []string          `json:"command,omitempty"`
	Cwd     string            `json:"cwd,omitempty"`
	Env     map[string]string `json:"env,omitempty"`
	Timeout *int              `json:"timeout,omitempty"`
}

type invocationConfig struct {
	Version      int    `json:"version"`
	Operation    string `json:"operation"`
	InvocationID string `json:"invocation_id"`
	Background   bool   `json:"background"`
}

type execution struct {
	done   chan struct{}
	cancel context.CancelFunc
	err    error // Published before closing done.
}

type processManager struct {
	mu      sync.Mutex
	store   invocationStore
	live    map[string]*execution
	outputs map[*outputFile]*os.File
	closed  bool
}

func newProcessManager(root string) (*processManager, error) {
	if err := os.MkdirAll(root, 0700); err != nil {
		return nil, err
	}
	return &processManager{
		store: invocationStore{root}, live: make(map[string]*execution),
		outputs: make(map[*outputFile]*os.File),
	}, nil
}

func (m *processManager) getLocked(id string) (invocationState, error) {
	result, err := m.store.read(id)
	if err == nil && result.Status == "interrupted" && m.live[id] != nil {
		result.Status = "in_progress"
	}
	return result, err
}

func (m *processManager) get(id string) (invocationState, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.getLocked(id)
}

func (m *processManager) start(req invocationRequest) (invocationState, *execution, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	id := req.Runtime.InvocationID
	current, err := m.getLocked(id)
	if err != nil || current.Status != "not_found" {
		// A repeated start never validates or executes the replacement command.
		return current, m.live[id], err
	}
	if m.closed {
		return invocationState{}, nil, errors.New("process manager is stopping")
	}
	timeout := 300
	if req.Timeout != nil {
		timeout = *req.Timeout
	}
	if len(req.Command) == 0 || req.Command[0] == "" || timeout < 1 || timeout > 3600 {
		return invocationState{}, nil, fmt.Errorf("%w: command must name an executable and timeout must be between 1 and 3600 seconds", errInvalidCommand)
	}
	for key := range req.Env {
		if key == "" || strings.ContainsAny(key, "=\x00") {
			return invocationState{}, nil, fmt.Errorf("%w: invalid environment variable name %q", errInvalidCommand, key)
		}
	}
	if err := m.store.start(id); err != nil {
		return invocationState{}, nil, err
	}
	// The execution belongs to the manager, not the HTTP request's context.
	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(timeout)*time.Second)
	run := &execution{done: make(chan struct{}), cancel: cancel}
	m.live[id] = run
	go m.run(ctx, req, run)
	return state(id, "in_progress"), run, nil
}

var errInvalidCommand = errors.New("invalid execution request")

func (m *processManager) busy() bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	return len(m.live) > 0
}

func (m *processManager) close() {
	m.mu.Lock()
	m.closed = true
	runs := make([]*execution, 0, len(m.live))
	for _, run := range m.live {
		run.cancel()
		runs = append(runs, run)
	}
	m.mu.Unlock()
	for _, run := range runs {
		<-run.done
	}
	// All executions have finished registering their output readers.
	m.mu.Lock()
	outputs := make(map[*outputFile]*os.File, len(m.outputs))
	for output, reader := range m.outputs {
		outputs[output] = reader
	}
	m.mu.Unlock()
	for _, reader := range outputs {
		reader.Close()
	}
	for output := range outputs {
		<-output.done
	}
}

func (m *processManager) run(ctx context.Context, req invocationRequest, run *execution) {
	defer run.cancel()
	id := req.Runtime.InvocationID
	result := state(id, "completed")
	output, err := m.execute(ctx, id, req)
	if err != nil {
		result.Error = &executionError{Code: "execution_failed", Message: err.Error()}
	} else {
		result.Result = output
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	// Keep this invocation live (and /ping busy) until publication finishes.
	run.err = m.store.finish(result)
	if run.err != nil {
		log.Printf("persist invocation %s: %v", id, run.err)
	}
	delete(m.live, id)
	close(run.done)
}

func (m *processManager) execute(ctx context.Context, id string, req invocationRequest) (*execResult, error) {
	stdout, stdoutWriter, err := m.capture(filepath.Join(m.store.dir(id), "stdout"))
	if err != nil {
		return nil, err
	}
	defer stdoutWriter.Close()
	stderr, stderrWriter, err := m.capture(filepath.Join(m.store.dir(id), "stderr"))
	if err != nil {
		return nil, err
	}
	defer stderrWriter.Close()

	cmd := exec.CommandContext(ctx, req.Command[0], req.Command[1:]...)
	cmd.Dir = req.Cwd
	if len(req.Env) > 0 {
		cmd.Env = cmd.Environ()
		for key, value := range req.Env {
			cmd.Env = append(cmd.Env, key+"="+value)
		}
	}
	// Own the pipes so Wait only reaps the direct process, without closing
	// readers still used by descendants or waiting for their output.
	cmd.Stdout, cmd.Stderr = stdoutWriter, stderrWriter
	err = cmd.Start()
	stdoutWriter.Close()
	stderrWriter.Close()
	if err != nil {
		return nil, fmt.Errorf("start command: %w", err)
	}

	err = cmd.Wait()
	timedOut := ctx.Err() == context.DeadlineExceeded
	var exitErr *exec.ExitError
	if err != nil && !errors.As(err, &exitErr) && !errors.Is(err, ctx.Err()) {
		return nil, fmt.Errorf("execute command: %w", err)
	}
	// One shared grace period for both streams; readers continue afterwards.
	grace, cancel := context.WithTimeout(context.Background(), outputGrace)
	defer cancel()
	for _, output := range []*outputFile{stdout, stderr} {
		select {
		case <-output.done:
		case <-grace.Done():
		}
	}
	out, outTruncated, outEOF, err := stdout.snapshot()
	if err != nil {
		return nil, err
	}
	errOut, errTruncated, errEOF, err := stderr.snapshot()
	if err != nil {
		return nil, err
	}
	return &execResult{
		ExitCode: cmd.ProcessState.ExitCode(), Stdout: out, Stderr: errOut,
		TimedOut: timedOut, OutputComplete: outEOF && errEOF,
		StdoutTruncated: outTruncated, StderrTruncated: errTruncated,
	}, nil
}

// capture owns the reader independently of the command and its deadline.
func (m *processManager) capture(path string) (*outputFile, *os.File, error) {
	output, err := newOutput(path)
	if err != nil {
		return nil, nil, err
	}
	reader, writer, err := os.Pipe()
	if err != nil {
		output.file.Close()
		return nil, nil, err
	}
	m.mu.Lock()
	m.outputs[output] = reader
	m.mu.Unlock()
	go func() {
		_, readErr := io.Copy(output, reader)
		reader.Close()
		output.mu.Lock()
		output.eof = readErr == nil
		if output.err == nil && !errors.Is(readErr, os.ErrClosed) {
			output.err = readErr
		}
		if err := output.file.Sync(); output.err == nil {
			output.err = err
		}
		if err := output.file.Close(); output.err == nil {
			output.err = err
		}
		output.closed = true
		if output.err != nil {
			log.Printf("capture output %s: %v", path, output.err)
		}
		output.mu.Unlock()
		m.mu.Lock()
		delete(m.outputs, output)
		m.mu.Unlock()
		close(output.done)
	}()
	return output, writer, nil
}

type outputFile struct {
	mu        sync.Mutex
	file      *os.File
	done      chan struct{}
	remaining int
	truncated bool
	eof       bool
	closed    bool
	err       error
}

func newOutput(path string) (*outputFile, error) {
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_EXCL, 0600)
	return &outputFile{file: f, done: make(chan struct{}), remaining: outputLimit}, err
}

func (w *outputFile) Write(p []byte) (int, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	size := len(p)
	if size > w.remaining {
		p = p[:w.remaining]
		w.truncated = true
	}
	n, err := w.file.Write(p)
	w.remaining -= n
	if err != nil {
		w.err = err
		return n, err
	}
	return size, nil
}

func (w *outputFile) snapshot() (string, bool, bool, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.err != nil {
		return "", w.truncated, w.eof, w.err
	}
	if !w.closed {
		if err := w.file.Sync(); err != nil {
			return "", w.truncated, w.eof, err
		}
	}
	data, err := os.ReadFile(w.file.Name())
	return string(data), w.truncated, w.eof, err
}
