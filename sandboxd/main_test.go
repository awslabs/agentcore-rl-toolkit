package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
)

func newTestServer(t *testing.T) (*httptest.Server, *serverState) {
	t.Helper()
	s := &serverState{}
	srv := httptest.NewServer(newMux(s))
	t.Cleanup(srv.Close)
	return srv, s
}

func getJSON(t *testing.T, url string) (int, map[string]string) {
	t.Helper()
	resp, err := http.Get(url)
	if err != nil {
		t.Fatalf("GET %s: %v", url, err)
	}
	defer resp.Body.Close()
	var body map[string]string
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	return resp.StatusCode, body
}

func postJSON(t *testing.T, url, payload string) (int, map[string]string) {
	t.Helper()
	resp, err := http.Post(url, "application/json", strings.NewReader(payload))
	if err != nil {
		t.Fatalf("POST %s: %v", url, err)
	}
	defer resp.Body.Close()
	var body map[string]string
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	return resp.StatusCode, body
}

func TestPingInitiallyHealthy(t *testing.T) {
	srv, _ := newTestServer(t)
	code, body := getJSON(t, srv.URL+"/ping")
	if code != http.StatusOK || body["status"] != "Healthy" {
		t.Fatalf("got %d %v, want 200 Healthy", code, body)
	}
	if _, ok := body["time_of_last_update"]; ok {
		t.Fatal("ping response must not include time_of_last_update")
	}
}

func TestStartFlipsToBusy(t *testing.T) {
	srv, _ := newTestServer(t)
	code, body := postJSON(t, srv.URL+"/invocations", `{"action":"start"}`)
	if code != http.StatusOK || body["status"] != "ok" || body["state"] != "busy" {
		t.Fatalf("got %d %v, want 200 ok/busy", code, body)
	}
	_, ping := getJSON(t, srv.URL+"/ping")
	if ping["status"] != "HealthyBusy" {
		t.Fatalf("ping after start = %v, want HealthyBusy", ping)
	}
}

func TestStopFlipsToHealthy(t *testing.T) {
	srv, _ := newTestServer(t)
	postJSON(t, srv.URL+"/invocations", `{"action":"start"}`)
	code, body := postJSON(t, srv.URL+"/invocations", `{"action":"stop"}`)
	if code != http.StatusOK || body["state"] != "healthy" {
		t.Fatalf("got %d %v, want 200 healthy", code, body)
	}
	_, ping := getJSON(t, srv.URL+"/ping")
	if ping["status"] != "Healthy" {
		t.Fatalf("ping after stop = %v, want Healthy", ping)
	}
}

func TestStatusDoesNotMutate(t *testing.T) {
	srv, s := newTestServer(t)
	code, body := postJSON(t, srv.URL+"/invocations", `{"action":"status"}`)
	if code != http.StatusOK || body["state"] != "healthy" {
		t.Fatalf("got %d %v, want 200 healthy", code, body)
	}
	if s.busy.Load() {
		t.Fatal("status action must not change state")
	}
	postJSON(t, srv.URL+"/invocations", `{"action":"start"}`)
	_, body = postJSON(t, srv.URL+"/invocations", `{"action":"status"}`)
	if body["state"] != "busy" {
		t.Fatalf("status after start = %v, want busy", body)
	}
	if !s.busy.Load() {
		t.Fatal("status action must not change state")
	}
}

func TestStartStopIdempotent(t *testing.T) {
	srv, _ := newTestServer(t)
	for i := 0; i < 2; i++ {
		code, body := postJSON(t, srv.URL+"/invocations", `{"action":"start"}`)
		if code != http.StatusOK || body["state"] != "busy" {
			t.Fatalf("start #%d: got %d %v", i+1, code, body)
		}
	}
	for i := 0; i < 2; i++ {
		code, body := postJSON(t, srv.URL+"/invocations", `{"action":"stop"}`)
		if code != http.StatusOK || body["state"] != "healthy" {
			t.Fatalf("stop #%d: got %d %v", i+1, code, body)
		}
	}
}

func TestUnknownFieldsIgnored(t *testing.T) {
	// Forward compatibility: a Phase 1 client sending ttl_seconds must work
	// against this binary.
	srv, _ := newTestServer(t)
	code, body := postJSON(t, srv.URL+"/invocations", `{"action":"start","ttl_seconds":60}`)
	if code != http.StatusOK || body["state"] != "busy" {
		t.Fatalf("got %d %v, want 200 busy", code, body)
	}
}

func TestBadRequests(t *testing.T) {
	srv, _ := newTestServer(t)
	cases := []struct {
		name    string
		payload string
	}{
		{"unknown action", `{"action":"reboot"}`},
		{"missing action", `{}`},
		{"malformed JSON", `{not json`},
	}
	for _, tc := range cases {
		code, body := postJSON(t, srv.URL+"/invocations", tc.payload)
		if code != http.StatusBadRequest || body["status"] != "error" {
			t.Fatalf("%s: got %d %v, want 400 error", tc.name, code, body)
		}
	}
}

func TestMethodNotAllowed(t *testing.T) {
	srv, _ := newTestServer(t)
	code, _ := postJSON(t, srv.URL+"/ping", `{}`)
	if code != http.StatusMethodNotAllowed {
		t.Fatalf("POST /ping: got %d, want 405", code)
	}
	code, _ = getJSON(t, srv.URL+"/invocations")
	if code != http.StatusMethodNotAllowed {
		t.Fatalf("GET /invocations: got %d, want 405", code)
	}
}

func TestConcurrentStartStop(t *testing.T) {
	srv, _ := newTestServer(t)
	var wg sync.WaitGroup
	for i := 0; i < 16; i++ {
		wg.Add(1)
		action := `{"action":"start"}`
		if i%2 == 0 {
			action = `{"action":"stop"}`
		}
		go func(payload string) {
			defer wg.Done()
			resp, err := http.Post(srv.URL+"/invocations", "application/json", strings.NewReader(payload))
			if err == nil {
				resp.Body.Close()
			}
		}(action)
	}
	wg.Wait()
	// State must be one of the two valid values; -race verifies no data race.
	_, ping := getJSON(t, srv.URL+"/ping")
	if ping["status"] != "Healthy" && ping["status"] != "HealthyBusy" {
		t.Fatalf("ping after concurrent ops = %v", ping)
	}
}
