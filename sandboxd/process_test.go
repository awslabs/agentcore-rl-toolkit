package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"
)

func request(id, command string, background bool) invocationRequest {
	return invocationRequest{
		Runtime: &invocationConfig{Version: 1, Operation: "start", InvocationID: id, Background: background},
		Command: command,
	}
}

func call(t *testing.T, url string, req invocationRequest) invocationState {
	t.Helper()
	body, err := json.Marshal(req)
	if err != nil {
		t.Fatal(err)
	}
	resp, err := http.Post(url+"/invocations", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var result invocationState
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("HTTP %d: %+v", resp.StatusCode, result)
	}
	return result
}

func waitFor(t *testing.T, condition func() bool) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for !condition() {
		if time.Now().After(deadline) {
			t.Fatal("condition did not become true")
		}
		time.Sleep(10 * time.Millisecond)
	}
}

func TestForegroundPersistsResult(t *testing.T) {
	srv, s := newTestServer(t)
	result := call(t, srv.URL, request("foreground", "printf hello; printf warning >&2; exit 3", false))
	if result.Status != "completed" || result.Result.ExitCode != 3 ||
		result.Result.Stdout != "hello" || result.Result.Stderr != "warning" || result.Result.TimedOut {
		t.Fatalf("unexpected result: %+v", result)
	}
	// A new manager can recover the terminal result using only the filesystem.
	restarted, err := newProcessManager(s.processes.store.root)
	if err != nil {
		t.Fatal(err)
	}
	recovered, err := restarted.get("foreground")
	if err != nil || recovered.Result == nil || *recovered.Result != *result.Result {
		t.Fatalf("recovered %+v, err %v", recovered, err)
	}
	started, err := os.ReadFile(filepath.Join(s.processes.store.dir("foreground"), "started.json"))
	if err != nil || bytes.Contains(started, []byte("printf")) {
		t.Fatalf("start record contains command or could not be read: %s, %v", started, err)
	}
}

func TestConcurrentStartsExecuteOnce(t *testing.T) {
	_, s := newTestServer(t)
	counter := filepath.Join(t.TempDir(), "count")
	req := request("retry", fmt.Sprintf("echo run >> %q; sleep 0.1", counter), true)
	var wg sync.WaitGroup
	for i := 0; i < 16; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, run, err := s.processes.start(req)
			if err != nil {
				t.Error(err)
			} else if run != nil {
				<-run.done
			}
		}()
	}
	wg.Wait()
	data, err := os.ReadFile(counter)
	if err != nil || string(data) != "run\n" {
		t.Fatalf("command ran more than once: %q, %v", data, err)
	}
	// Retry validation must not evaluate the replacement payload.
	badTimeout := -1
	req.Command, req.Timeout = "", &badTimeout
	result, _, err := s.processes.start(req)
	if err != nil || result.Status != "completed" {
		t.Fatalf("retry: %+v, %v", result, err)
	}
}

func TestIdenticalCommandsWithDistinctIDs(t *testing.T) {
	srv, _ := newTestServer(t)
	counter := filepath.Join(t.TempDir(), "count")
	command := fmt.Sprintf("echo run >> %q", counter)
	call(t, srv.URL, request("one", command, false))
	call(t, srv.URL, request("two", command, false))
	data, _ := os.ReadFile(counter)
	if string(data) != "run\nrun\n" {
		t.Fatalf("distinct invocations did not both execute: %q", data)
	}
}

func TestDisconnectDoesNotCancelExecution(t *testing.T) {
	srv, s := newTestServer(t)
	marker := filepath.Join(t.TempDir(), "started")
	release := filepath.Join(t.TempDir(), "release")
	req := request("disconnect", fmt.Sprintf(
		"echo started > %q; while [ ! -e %q ]; do sleep 0.01; done; printf finished", marker, release,
	), false)
	data, _ := json.Marshal(req)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	httpReq, _ := http.NewRequestWithContext(ctx, "POST", srv.URL+"/invocations", bytes.NewReader(data))
	disconnected := make(chan struct{})
	go func() {
		defer close(disconnected)
		resp, err := http.DefaultClient.Do(httpReq)
		if err == nil {
			resp.Body.Close()
		}
	}()
	waitFor(t, func() bool {
		_, err := os.Stat(marker)
		return err == nil
	})
	cancel()
	<-disconnected
	_, released := postJSON(t, srv.URL+"/invocations", `{"action":"stop"}`)
	if released["state"] != "busy" {
		t.Fatalf("releasing a hold must not idle a running command: %v", released)
	}
	_, ping := getJSON(t, srv.URL+"/ping")
	if ping["status"] != "HealthyBusy" {
		t.Fatalf("execution must keep session busy: %v", ping)
	}
	if err := os.WriteFile(release, nil, 0600); err != nil {
		t.Fatal(err)
	}
	var result invocationState
	waitFor(t, func() bool {
		result, _ = s.processes.get("disconnect")
		return result.Status == "completed"
	})
	if result.Result == nil || result.Result.Stdout != "finished" {
		t.Fatalf("lost output after disconnect: %+v", result)
	}
	_, ping = getJSON(t, srv.URL+"/ping")
	if ping["status"] != "Healthy" {
		t.Fatalf("unheld session should become idle: %v", ping)
	}
}

func TestWaitsForDescendantOutput(t *testing.T) {
	for _, exitCode := range []int{0, 7} {
		t.Run(strconv.Itoa(exitCode), func(t *testing.T) {
			srv, _ := newTestServer(t)
			command := fmt.Sprintf(
				"(sleep 2; printf late; printf warning >&2) & printf parent-done; exit %d", exitCode,
			)
			result := call(t, srv.URL, request("output-wait", command, false)).Result
			if result == nil || result.ExitCode != exitCode || result.TimedOut ||
				result.Stdout != "parent-donelate" || result.Stderr != "warning" {
				t.Fatalf("lost descendant output or shell exit code: %+v", result)
			}
		})
	}
}

func TestChildSurvivesCompletionAndTimeout(t *testing.T) {
	for _, tc := range []struct {
		name     string
		command  string
		exitCode int
		timedOut bool
	}{
		{"redirected-output", "sleep 30 >/dev/null 2>&1 & echo $!", 0, false},
		{"timeout-running-shell", "sleep 30 & echo $!; wait", -1, true},
		{"timeout-after-shell-exit", "sleep 30 & echo $!; exit 7", 7, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			srv, _ := newTestServer(t)
			req := request(tc.name, "printf partial >&2; "+tc.command, false)
			timeout := 2
			req.Timeout = &timeout
			start := time.Now()
			result := call(t, srv.URL, req).Result
			if result == nil {
				t.Fatal("expected an execution result")
			}
			pid, err := strconv.Atoi(strings.TrimSpace(result.Stdout))
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { _ = syscall.Kill(pid, syscall.SIGKILL) })
			if result.TimedOut != tc.timedOut || result.ExitCode != tc.exitCode || result.Stderr != "partial" {
				t.Fatalf("unexpected result: %+v", result)
			}
			if time.Since(start) > 5*time.Second {
				t.Fatal("output wait exceeded the execution deadline")
			}
			// kill(pid, 0) also succeeds for zombies; verify the child is still alive.
			stat, err := os.ReadFile(fmt.Sprintf("/proc/%d/stat", pid))
			if err != nil || strings.Contains(string(stat), ") Z ") {
				t.Fatalf("child did not survive: %s, %v", stat, err)
			}
		})
	}
}

func TestOutputEOFBeforeProcessExit(t *testing.T) {
	for _, tc := range []struct {
		name     string
		command  string
		exitCode int
		timedOut bool
	}{
		{"normal-exit", "sleep 0.1; exit 7", 7, false},
		{"timeout", "exec sleep 30", -1, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			srv, _ := newTestServer(t)
			req := request(tc.name, "printf before; exec 1>&- 2>&-; "+tc.command, false)
			timeout := 1
			req.Timeout = &timeout
			result := call(t, srv.URL, req).Result
			if result == nil || result.ExitCode != tc.exitCode || result.TimedOut != tc.timedOut ||
				result.Stdout != "before" {
				t.Fatalf("unexpected result after output EOF: %+v", result)
			}
		})
	}
}

func TestMissingAndInterruptedNeverExecute(t *testing.T) {
	srv, s := newTestServer(t)
	marker := filepath.Join(t.TempDir(), "must-not-exist")
	req := request("missing", fmt.Sprintf("touch %q", marker), false)
	req.Runtime.Operation = "get"
	if result := call(t, srv.URL, req); result.Status != "not_found" {
		t.Fatalf("missing: %+v", result)
	}
	if err := s.processes.store.start("stale"); err != nil {
		t.Fatal(err)
	}
	req.Runtime.InvocationID = "stale"
	if result := call(t, srv.URL, req); result.Status != "interrupted" {
		t.Fatalf("stale get: %+v", result)
	}
	req.Runtime.Operation = "start"
	if result := call(t, srv.URL, req); result.Status != "interrupted" {
		t.Fatalf("stale start: %+v", result)
	}
	if _, err := os.Stat(marker); !os.IsNotExist(err) {
		t.Fatal("get or interrupted start ran the command")
	}
}

func TestOutputLimitDrainsAndMarksTruncation(t *testing.T) {
	srv, s := newTestServer(t)
	result := call(t, srv.URL, request("output", "head -c 300000 /dev/zero; printf done >&2", false)).Result
	if result == nil || result.ExitCode != 0 || len(result.Stdout) != outputLimit ||
		!result.StdoutTruncated || result.StderrTruncated || result.Stderr != "done" {
		t.Fatalf("output limit failed: %+v", result)
	}
	info, err := os.Stat(filepath.Join(s.processes.store.dir("output"), "stdout"))
	if err != nil || info.Size() != outputLimit {
		t.Fatalf("output file not bounded: %v, %v", info, err)
	}
}

func TestPersistenceFailureCannotReportSuccess(t *testing.T) {
	srv, s := newTestServer(t)
	release := filepath.Join(t.TempDir(), "release")
	req := request("persist-failure", fmt.Sprintf("while [ ! -e %q ]; do sleep 0.01; done", release), true)
	call(t, srv.URL, req)
	// Make atomic result publication fail even though the command succeeds.
	if err := os.Mkdir(filepath.Join(s.processes.store.dir("persist-failure"), "result.json"), 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(release, nil, 0600); err != nil {
		t.Fatal(err)
	}
	waitFor(t, func() bool { return !s.processes.busy() })
	_, err := s.processes.get("persist-failure")
	if err == nil {
		t.Fatal("get must fail when terminal publication failed")
	}
}

func TestStartRecordFailureDoesNotExecute(t *testing.T) {
	_, s := newTestServer(t)
	marker := filepath.Join(t.TempDir(), "must-not-exist")
	// An incomplete, already-claimed directory cannot be silently reused.
	if err := os.Mkdir(s.processes.store.dir("blocked"), 0700); err != nil {
		t.Fatal(err)
	}
	_, run, err := s.processes.start(request("blocked", fmt.Sprintf("touch %q", marker), true))
	if err == nil || run != nil {
		t.Fatal("expected failed claim")
	}
	if _, err := os.Stat(marker); !os.IsNotExist(err) {
		t.Fatal("command ran before a start record was persisted")
	}
}

func TestSpawnFailureIsPersisted(t *testing.T) {
	srv, _ := newTestServer(t)
	req := request("spawn-failure", "echo hi", false)
	req.Shell = "/nonexistent-shell"
	result := call(t, srv.URL, req)
	if result.Status != "completed" || result.Error == nil || result.Result != nil {
		t.Fatalf("spawn failure: %+v", result)
	}
}

func TestOutputFailureWithNonzeroExit(t *testing.T) {
	// exec.Cmd.Wait can prioritize an exit error over an output-copy error.
	// A failed log write must still be surfaced, not returned as complete output.
	full, err := os.OpenFile("/dev/full", os.O_WRONLY, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer full.Close()
	output := &outputFile{file: full, remaining: outputLimit}
	cmd := exec.Command("/bin/sh", "-c", "printf data; exit 3")
	cmd.Stdout = output
	var exitErr *exec.ExitError
	if err := cmd.Run(); !errors.As(err, &exitErr) || exitErr.ExitCode() != 3 {
		t.Fatalf("expected command exit 3, got %v", err)
	}
	// Sync on /dev/full also fails, but that must not mask the original ENOSPC.
	if _, err := output.finish(); !errors.Is(err, syscall.ENOSPC) {
		t.Fatalf("expected the original disk-full write error, got %v", err)
	}
}

func TestInvalidExecutionRequests(t *testing.T) {
	srv, _ := newTestServer(t)
	for _, payload := range []string{
		`{"_agentcore_runtime":{"version":1,"operation":"get","invocation_id":"../escape"}}`,
		`{"_agentcore_runtime":{"version":2,"operation":"get","invocation_id":"test"}}`,
		`{"_agentcore_runtime":{"version":1,"operation":"delete","invocation_id":"test"}}`,
		`{"_agentcore_runtime":{"version":1,"operation":"start","invocation_id":"test"},"command":"echo hi","timeout":0}`,
	} {
		code, _ := postJSON(t, srv.URL+"/invocations", payload)
		if code != http.StatusBadRequest {
			t.Fatalf("got HTTP %d for %s", code, payload)
		}
	}
}

func TestSessionHoldSurvivesCommandCompletion(t *testing.T) {
	srv, _ := newTestServer(t)
	postJSON(t, srv.URL+"/invocations", `{"action":"start"}`)
	call(t, srv.URL, request("hold", "true", false))
	_, ping := getJSON(t, srv.URL+"/ping")
	if ping["status"] != "HealthyBusy" {
		t.Fatal("finishing a command must not release the session hold")
	}
}

func TestForegroundPublicationFailure(t *testing.T) {
	_, s := newTestServer(t)
	scratch := t.TempDir()
	marker, release := filepath.Join(scratch, "started"), filepath.Join(scratch, "release")
	req := request("foreground-failure", fmt.Sprintf(
		"echo started > %q; while [ ! -e %q ]; do sleep 0.01; done", marker, release,
	), false)
	recorder := httptest.NewRecorder()
	done := make(chan struct{})
	go func() {
		defer close(done)
		s.handleExecution(recorder, httptest.NewRequest("POST", "/invocations", nil), req)
	}()
	// The request must register and launch the command before publication fails.
	waitFor(t, func() bool {
		_, err := os.Stat(marker)
		return err == nil
	})
	if err := os.Mkdir(filepath.Join(s.processes.store.dir("foreground-failure"), "result.json"), 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(release, nil, 0600); err != nil {
		t.Fatal(err)
	}
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("foreground request did not finish after command release")
	}
	if recorder.Code != http.StatusInternalServerError {
		t.Fatalf("publication failure returned HTTP %d", recorder.Code)
	}
}
