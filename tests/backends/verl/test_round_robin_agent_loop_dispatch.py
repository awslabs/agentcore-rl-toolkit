import sys
from types import ModuleType, SimpleNamespace

from agentcore_rl_toolkit.backends.verl.trainer_mixins import round_robin_agent_loop_dispatch


class _RemoteMethod:
    def __init__(self, worker_name, calls):
        self.worker_name = worker_name
        self.calls = calls

    def remote(self, chunk):
        self.calls.append((self.worker_name, chunk))
        return self.worker_name, chunk


def _worker(worker_name, calls):
    return SimpleNamespace(generate_sequences=_RemoteMethod(worker_name, calls))


def _prompts(*chunks):
    return SimpleNamespace(chunk=lambda _: chunks)


def _upstream_generate_sequences(manager, prompts):
    for worker, chunk in zip(manager.agent_loop_workers, prompts.chunk(len(manager.agent_loop_workers)), strict=False):
        worker.generate_sequences.remote(chunk)


def _install_fake_verl_agent_loop_module(monkeypatch):
    """Provide only the class the installer imports, without importing verl/Ray."""
    verl = ModuleType("verl")
    trainer = ModuleType("verl.trainer")
    ppo = ModuleType("verl.trainer.ppo")
    v1 = ModuleType("verl.trainer.ppo.v1")
    agent_loop_tq = ModuleType("verl.trainer.ppo.v1.agent_loop_tq")

    class AgentLoopManagerTQ:
        generate_sequences = _upstream_generate_sequences

    agent_loop_tq.AgentLoopManagerTQ = AgentLoopManagerTQ
    verl.trainer = trainer
    trainer.ppo = ppo
    ppo.v1 = v1
    v1.agent_loop_tq = agent_loop_tq
    monkeypatch.setitem(sys.modules, "verl", verl)
    monkeypatch.setitem(sys.modules, "verl.trainer", trainer)
    monkeypatch.setitem(sys.modules, "verl.trainer.ppo", ppo)
    monkeypatch.setitem(sys.modules, "verl.trainer.ppo.v1", v1)
    monkeypatch.setitem(sys.modules, "verl.trainer.ppo.v1.agent_loop_tq", agent_loop_tq)
    return AgentLoopManagerTQ


def test_singleton_dispatch_rotates_workers():
    calls = []
    manager = SimpleNamespace(agent_loop_workers=[_worker("zero", calls), _worker("one", calls), _worker("two", calls)])

    for chunk in ("first", "second", "third", "fourth"):
        round_robin_agent_loop_dispatch._generate_sequences_round_robin(
            manager, _prompts(chunk), _upstream_generate_sequences
        )

    assert calls == [("zero", "first"), ("one", "second"), ("two", "third"), ("zero", "fourth")]


def test_multi_chunk_dispatch_advances_by_dispatched_worker_count():
    calls = []
    manager = SimpleNamespace(agent_loop_workers=[_worker("zero", calls), _worker("one", calls), _worker("two", calls)])

    round_robin_agent_loop_dispatch._generate_sequences_round_robin(
        manager, _prompts("first", "second"), _upstream_generate_sequences
    )
    round_robin_agent_loop_dispatch._generate_sequences_round_robin(
        manager, _prompts("third"), _upstream_generate_sequences
    )

    assert calls == [("zero", "first"), ("one", "second"), ("two", "third")]


def test_install_replaces_the_v1_tq_dispatcher_once(monkeypatch):
    calls = []
    manager = SimpleNamespace(agent_loop_workers=[_worker("zero", calls), _worker("one", calls)])
    AgentLoopManagerTQ = _install_fake_verl_agent_loop_module(monkeypatch)

    round_robin_agent_loop_dispatch.install_round_robin_agent_loop_dispatch()
    patched_generate_sequences = AgentLoopManagerTQ.generate_sequences
    patched_generate_sequences(manager, _prompts("first"))
    patched_generate_sequences(manager, _prompts("second"))
    round_robin_agent_loop_dispatch.install_round_robin_agent_loop_dispatch()

    assert patched_generate_sequences is AgentLoopManagerTQ.generate_sequences
    assert patched_generate_sequences._agentcore_round_robin_dispatch
    assert calls == [("zero", "first"), ("one", "second")]
