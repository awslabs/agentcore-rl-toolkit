// agentcore-sandboxd is a minimal health shim that makes arbitrary Docker images
// satisfy the Bedrock AgentCore Runtime container contract (port 8080, /ping,
// /invocations) so they can be used as sandboxes for command execution.
//
// It holds a single busy/healthy flag driven by /invocations actions. Command
// execution itself is NOT handled here — the SDK uses AgentCore Runtime's native
// InvokeAgentRuntimeCommand API, which runs shell commands in this container
// out-of-band of this server.
package main

import (
	"encoding/json"
	"log"
	"net/http"
	"sync/atomic"
)

// serverState holds the busy flag reported via /ping. While busy, AgentCore
// keeps the runtime session alive past its idle timeout ("HealthyBusy").
type serverState struct {
	busy atomic.Bool
}

// invocationRequest is the /invocations payload. Unknown JSON fields are
// deliberately ignored so future clients (e.g. sending ttl_seconds) remain
// compatible with this binary.
type invocationRequest struct {
	Action string `json:"action"`
}

func (s *serverState) stateName() string {
	if s.busy.Load() {
		return "busy"
	}
	return "healthy"
}

func writeJSON(w http.ResponseWriter, status int, body map[string]string) {
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
	if s.busy.Load() {
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
	switch req.Action {
	case "start":
		s.busy.Store(true)
		log.Printf("sandbox started (state=busy)")
		writeJSON(w, http.StatusOK, map[string]string{"status": "ok", "state": "busy"})
	case "stop":
		s.busy.Store(false)
		log.Printf("sandbox stopped (state=healthy)")
		writeJSON(w, http.StatusOK, map[string]string{"status": "ok", "state": "healthy"})
	case "status":
		writeJSON(w, http.StatusOK, map[string]string{"status": "ok", "state": s.stateName()})
	case "":
		writeJSON(w, http.StatusBadRequest, map[string]string{"status": "error", "error": "missing action"})
	default:
		writeJSON(w, http.StatusBadRequest, map[string]string{"status": "error", "error": "unknown action: " + req.Action})
	}
}

func newMux(s *serverState) *http.ServeMux {
	mux := http.NewServeMux()
	mux.HandleFunc("/ping", s.handlePing)
	mux.HandleFunc("/invocations", s.handleInvocations)
	return mux
}

func main() {
	s := &serverState{}
	addr := "0.0.0.0:8080"
	log.Printf("agentcore-sandboxd listening on %s", addr)
	if err := http.ListenAndServe(addr, newMux(s)); err != nil {
		log.Fatalf("server failed: %v", err)
	}
}
