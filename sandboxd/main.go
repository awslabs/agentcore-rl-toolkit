// agentcore-sandboxd manages sandbox sessions and recoverable command executions
// through the AgentCore Runtime container contract (/ping and /invocations).
package main

import (
	"encoding/json"
	"errors"
	"flag"
	"log"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"sync/atomic"
	"syscall"
)

// serverState holds the busy flag reported via /ping. While busy, AgentCore
// keeps the runtime session alive past its idle timeout ("HealthyBusy").
type serverState struct {
	busy      atomic.Bool
	processes *processManager
}

func (s *serverState) isBusy() bool {
	return s.busy.Load() || s.processes.busy()
}

func (s *serverState) stateName() string {
	if s.isBusy() {
		return "busy"
	}
	return "healthy"
}

func writeJSON(w http.ResponseWriter, status int, body any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(body)
}

func (s *serverState) handlePing(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"status": "error", "error": "method not allowed"})
		return
	}
	// Never include time_of_last_update: advancing it on every ping would
	// prevent the session from ever idling out.
	status := "Healthy"
	if s.isBusy() {
		status = "HealthyBusy"
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": status})
}

func (s *serverState) handleInvocations(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"status": "error", "error": "method not allowed"})
		return
	}
	var req invocationRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"status": "error", "error": "malformed JSON body"})
		return
	}
	if req.Runtime != nil {
		s.handleExecution(w, r, req)
		return
	}
	switch req.Action {
	case "start":
		s.busy.Store(true)
		log.Printf("sandbox started (state=busy)")
		writeJSON(w, http.StatusOK, map[string]string{"status": "ok", "state": "busy"})
	case "stop":
		s.busy.Store(false)
		log.Printf("sandbox session hold released")
		writeJSON(w, http.StatusOK, map[string]string{"status": "ok", "state": s.stateName()})
	case "status":
		writeJSON(w, http.StatusOK, map[string]string{"status": "ok", "state": s.stateName()})
	case "":
		writeJSON(w, http.StatusBadRequest, map[string]string{"status": "error", "error": "missing action"})
	default:
		writeJSON(w, http.StatusBadRequest, map[string]string{"status": "error", "error": "unknown action: " + req.Action})
	}
}

func (s *serverState) handleExecution(w http.ResponseWriter, r *http.Request, req invocationRequest) {
	config := req.Runtime
	if config.Version != 1 || !invocationIDPattern.MatchString(config.InvocationID) {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "expected version 1 and a valid invocation_id"})
		return
	}
	var result invocationState
	var err error
	switch config.Operation {
	case "get":
		result, err = s.processes.get(config.InvocationID)
	case "start":
		var run *execution
		result, run, err = s.processes.start(req)
		if err == nil && run != nil && !config.Background {
			select {
			case <-r.Context().Done():
				return // Only stop waiting. The manager still owns the process.
			case <-run.done:
				err = run.err
				if err == nil {
					result, err = s.processes.get(config.InvocationID)
				}
			}
		}
	default:
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "unknown invocation operation"})
		return
	}
	if err != nil {
		status := http.StatusInternalServerError
		if errors.Is(err, errInvalidCommand) {
			status = http.StatusBadRequest
		}
		writeJSON(w, status, map[string]string{"error": err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, result)
}

func newMux(s *serverState) *http.ServeMux {
	mux := http.NewServeMux()
	mux.HandleFunc("/ping", s.handlePing)
	mux.HandleFunc("/invocations", s.handleInvocations)
	return mux
}

func main() {
	root := flag.String("state-dir", filepath.Join(os.TempDir(), "agentcore-sandboxd"), "local invocation record directory")
	addr := flag.String("listen", "0.0.0.0:8080", "HTTP listen address")
	flag.Parse()
	processes, err := newProcessManager(*root)
	if err != nil {
		log.Fatal(err)
	}
	s := &serverState{processes: processes}
	server := &http.Server{Addr: *addr, Handler: newMux(s)}
	stop := make(chan os.Signal, 1)
	signal.Notify(stop, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-stop
		processes.close()
		_ = server.Close()
	}()
	log.Printf("agentcore-sandboxd listening on %s", *addr)
	if err := server.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		log.Fatalf("server failed: %v", err)
	}
}
