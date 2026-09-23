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
	"sync"
	"time"
)

const outputLimit = 256 * 1024 // Per stream; continue draining after this limit.

var invocationIDPattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$`)

type invocationRequest struct {
	Action  string            `json:"action,omitempty"`
	Runtime *invocationConfig `json:"_agentcore_runtime,omitempty"`
	Command string            `json:"command,omitempty"`
	Shell   string            `json:"shell,omitempty"`
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
	mu     sync.Mutex
	store  invocationStore
	live   map[string]*execution
	closed bool
}

func newProcessManager(root string) (*processManager, error) {
	if err := os.MkdirAll(root, 0700); err != nil {
		return nil, err
	}
	return &processManager{store: invocationStore{root}, live: make(map[string]*execution)}, nil
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
	if req.Command == "" || timeout < 1 || timeout > 3600 {
		return invocationState{}, nil, errInvalidCommand
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

var errInvalidCommand = errors.New("command must be nonempty and timeout must be between 1 and 3600 seconds")

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
	stdout, err := newOutput(filepath.Join(m.store.dir(id), "stdout"))
	if err != nil {
		return nil, err
	}
	defer stdout.file.Close()
	stderr, err := newOutput(filepath.Join(m.store.dir(id), "stderr"))
	if err != nil {
		return nil, err
	}
	defer stderr.file.Close()

	shell := req.Shell
	if shell == "" {
		shell = "/bin/sh"
	}
	cmd := exec.CommandContext(ctx, shell, "-c", req.Command)
	stdoutPipe, err := cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}
	defer stdoutPipe.Close()
	stderrPipe, err := cmd.StderrPipe()
	if err != nil {
		return nil, err
	}
	defer stderrPipe.Close()
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("start command: %w", err)
	}

	outCtx, outCancel := context.WithCancel(ctx)
	defer outCancel()
	var readers sync.WaitGroup
	copyOutput := func(dst *outputFile, src io.Reader) {
		defer readers.Done()
		if _, err := io.Copy(dst, src); err != nil && !errors.Is(err, os.ErrClosed) {
			dst.err = err
		}
	}
	readers.Add(2)
	go copyOutput(stdout, stdoutPipe)
	go copyOutput(stderr, stderrPipe)
	go func() {
		readers.Wait()
		outCancel()
	}()

	// Descendants may hold the pipes after the shell exits. Wait for EOF or
	// cancellation before Wait closes the readers and reaps the direct process.
	<-outCtx.Done()
	err = cmd.Wait()
	readers.Wait()
	timedOut := ctx.Err() == context.DeadlineExceeded
	var exitErr *exec.ExitError
	if err != nil && !errors.As(err, &exitErr) && !errors.Is(err, ctx.Err()) {
		return nil, fmt.Errorf("execute command: %w", err)
	}
	out, err := stdout.finish()
	if err != nil {
		return nil, err
	}
	errOut, err := stderr.finish()
	if err != nil {
		return nil, err
	}
	return &execResult{
		ExitCode: cmd.ProcessState.ExitCode(), Stdout: out, Stderr: errOut,
		TimedOut:        timedOut,
		StdoutTruncated: stdout.truncated, StderrTruncated: stderr.truncated,
	}, nil
}

type outputFile struct {
	file      *os.File
	remaining int
	truncated bool
	err       error
}

func newOutput(path string) (*outputFile, error) {
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_EXCL, 0600)
	return &outputFile{file: f, remaining: outputLimit}, err
}

func (w *outputFile) Write(p []byte) (int, error) {
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

func (w *outputFile) finish() (string, error) {
	if w.err != nil {
		return "", w.err
	}
	if err := w.file.Sync(); err != nil {
		return "", err
	}
	data, err := os.ReadFile(w.file.Name())
	return string(data), err
}
