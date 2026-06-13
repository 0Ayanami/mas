"""Distributed multi-agent memory consensus simulation."""

from __future__ import annotations
import os, sys, json, math, types
from dataclasses import dataclass, field
import importlib, importlib.util

SRC = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, SRC)

# --- Bootstrap mas_framework as a dummy package ---
# This avoids importing __init__.py (which triggers the CAMEL chain).
_my_framework = types.ModuleType("mas_framework")
_my_framework.__path__ = [os.path.join(SRC, "mas_framework")]
_my_framework.__file__ = os.path.join(SRC, "mas_framework", "__init__.py")
_my_framework.__package__ = "mas_framework"
sys.modules["mas_framework"] = _my_framework

# Load models.py as mas_framework.models
_models_path = os.path.join(SRC, "mas_framework", "models.py")
_m_spec = importlib.util.spec_from_file_location("mas_framework.models", _models_path)
_models = importlib.util.module_from_spec(_m_spec)
sys.modules["mas_framework.models"] = _models
_m_spec.loader.exec_module(_models)

# Load consensus.py as mas_framework.consensus
_cons_path = os.path.join(SRC, "mas_framework", "consensus.py")
_c_spec = importlib.util.spec_from_file_location("mas_framework.consensus", _cons_path)
_consensus = importlib.util.module_from_spec(_c_spec)
sys.modules["mas_framework.consensus"] = _consensus
_c_spec.loader.exec_module(_consensus)

MemoryProposal = _models.MemoryProposal
ProposalStatus = _models.ProposalStatus
VerificationVector = _models.VerificationVector
SelfVerification = _models.SelfVerification
MultiAgentVerificationSummary = _models.MultiAgentVerificationSummary
AgentState = _models.AgentState
ConsensusDecision = _models.ConsensusDecision
SmartQuorumPolicy = _consensus.SmartQuorumPolicy


class MockMemory:
    def __init__(self):
        self._store = {}
    def add_proposal(self, proposal, user_id):
        k = f"{proposal.proposal_id}:{user_id}"
        self._store[k] = proposal
        return {"success": True, "id": k}
    def memory_count(self):
        return len(self._store)


@dataclass
class MockAgent:
    agent_id: str
    base_accuracy: float = 0.85
    bias_toward_accept: float = 0.0
    conf_threshold: float = 0.7
    initial_weight: float = 1.0
    memory: MockMemory = field(default_factory=MockMemory)
    state: AgentState = field(default_factory=AgentState)

    def __post_init__(self):
        self.state.weight = self.initial_weight

    def verify(self, proposal):
        import random
        rng = random.Random(self.agent_id + proposal.proposal_id)
        sv = proposal.verification.self_verification
        v = rng.random() < self.base_accuracy if sv.veracity == 1 else rng.random() >= self.base_accuracy
        r = rng.random() < self.base_accuracy if sv.rationality == 1 else rng.random() >= self.base_accuracy
        val = rng.random() < self.base_accuracy if sv.value == 1 else rng.random() >= self.base_accuracy
        s = rng.random() < self.base_accuracy if sv.security == 1 else rng.random() >= self.base_accuracy
        agent_conf = max(0.0, min(1.0, self.base_accuracy * proposal.self_confidence + self.bias_toward_accept))
        vr = agent_conf >= self.conf_threshold
        parts = []
        if v: parts.append("v OK")
        if r: parts.append("r OK")
        if val: parts.append("val OK")
        if s: parts.append("s OK")
        return VerificationVector(
            veracity=int(v), rationality=int(r), value=int(val), security=int(s),
            confidence=round(agent_conf, 4),
            rationale="; ".join(parts) if parts else "all failed",
            verifier_id=self.agent_id, conf_threshold=self.conf_threshold,
            vote_result=vr, weight=self.state.weight,
        )

    def self_verify(self, proposal):
        vv = self.verify(proposal)
        if vv.vote_result:
            proposal.verifications.append(vv)
            proposal.verification.self_verification = SelfVerification(
                veracity=vv.veracity, rationality=vv.rationality,
                value=vv.value, security=vv.security,
                confidence=vv.confidence, rationale=vv.rationale,
            )
        return vv

    def update_state(self, mac, status):
        self.state.proposal_sum += 1
        if status == ProposalStatus.ACCEPTED:
            self.state.proposal_submitted += 1
        if self.state.proposal_sum > 0:
            self.state.historical_conf = round(self.state.proposal_submitted / self.state.proposal_sum, 4)
        if mac and mac.confidence is not None:
            self.state.verified_conf_sum += mac.confidence
            self.state.verified_conf = (
                round(self.state.verified_conf_sum / self.state.proposal_sum, 4)
                if self.state.proposal_sum > 0 else 0.0
            )
        self.state.weight = round(
            math.exp(
                (0.5 * (self.state.verified_conf or 0) + 0.5 * (self.state.historical_conf or 0))
                * (self.state.base or 1.0)
            ),
            4,
        )


class SimOrch:
    def __init__(self, agents, policy=None):
        self.agents = {a.agent_id: a for a in agents}
        self.policy = policy or SmartQuorumPolicy()

    def run_round(self, proposal):
        pid = proposal.agent_id
        print("  [1] " + pid + " self-verify")
        self.agents[pid].self_verify(proposal)
        vs = [a for i, a in self.agents.items() if i != pid]
        print("  [2] " + str(len(vs)) + " validators: " + str([v.agent_id for v in vs]))
        for v in vs:
            vv = v.verify(proposal)
            proposal.verifications.append(vv)
            print(
                "      " + v.agent_id + ": v=" + str(vv.veracity)
                + " r=" + str(vv.rationality) + " val=" + str(vv.value)
                + " s=" + str(vv.security) + " c=" + str(round(vv.confidence, 3))
                + " vote=" + str(vv.vote_result)
            )
        print("  [3] Consensus")
        proposal.consensus_round += 1
        d = self.policy.decide(proposal, agent_count=len(self.agents))
        proposal.status = d.status
        print(
            "      " + d.status.value + " pos=" + str(d.positive_votes)
            + " thresh=" + str(d.threshold) + " avg_c=" + str(d.average_confidence)
        )
        if d.status == ProposalStatus.ACCEPTED:
            for a in self.agents.values():
                a.memory.add_proposal(proposal, user_id=a.agent_id)
            print("  [4] Committed to " + str(len(self.agents)) + " memories")
        mac = proposal.verification.multi_agent_verification
        for a in self.agents.values():
            a.update_state(mac, d.status)
        print("  [5] States updated")
        if mac:
            print(
                "      MA: v=" + str(mac.veracity) + " r=" + str(mac.rationality)
                + " val=" + str(mac.value) + " s=" + str(mac.security)
                + " c=" + str(mac.confidence)
            )
        return d


def mkprop(agent="r1", conf=0.8, v=1, r=1, val=1, s=1, title="T"):
    p = MemoryProposal(
        agent_id=agent, task_id="t", memory_type="research_note", title=title,
        thoughts_decision="x", action=[], data={}, results_observation=[],
        self_confidence=conf,
    )
    p.verification.self_verification = SelfVerification(
        veracity=v, rationality=r, value=val, security=s,
        confidence=conf, rationale="sc",
    )
    return p


def mkagents(n=4, acc=0.85, bias=0.0, w=1.0):
    rs = ["r1", "m2", "s3", "s4"]
    return [MockAgent(rs[i] if i < len(rs) else "a" + str(i), acc, bias, 0.7, w) for i in range(n)]


# === Scenarios ===

def test_unanimous():
    print("\n=== S1: Unanimous Acceptance ===")
    a = mkagents(4, 0.95, 0.05)
    d = SimOrch(a).run_round(mkprop(conf=0.9))
    assert d.status == ProposalStatus.ACCEPTED
    for ag in a:
        assert ag.memory.memory_count() > 0
    print("  PASS: " + d.status.value)


def test_rejection():
    print("\n=== S2: Majority Rejection ===")
    a = mkagents(4, 0.30, -0.2)
    d = SimOrch(a).run_round(mkprop(conf=0.6, val=0))
    assert d.status == ProposalStatus.REJECTED
    print("  PASS: " + d.status.value)


def test_mixed():
    print("\n=== S3: Mixed (Weighted Vote) ===")
    agents = [
        MockAgent("r1", 0.95, 0.05, 0.7, 1.0),
        MockAgent("m2", 0.70, -0.10, 0.7, 1.2),
        MockAgent("s3", 0.60, 0.00, 0.7, 0.9),
        MockAgent("s4", 0.50, -0.20, 0.7, 1.1),
    ]
    d = SimOrch(agents).run_round(mkprop(conf=0.75, s=0))
    print("  PASS: " + d.status.value + " pos=" + str(d.positive_votes) + " thresh=" + str(d.threshold))


def test_multiround():
    print("\n=== S4: Multi-Round ===")
    a = mkagents(4, 0.82)
    o = SimOrch(a)
    n_acc = 0
    rounds = [
        ("R1 - Init", 0.85, 1, 1, 1, 1, "r1"),
        ("R2 - Evid", 0.70, 1, 1, 0, 1, "m2"),
        ("R3 - Final", 0.90, 1, 1, 1, 1, "r1"),
    ]
    for title, cf, v, r, val, s, pid in rounds:
        print("\n  -- " + title + " --")
        d = o.run_round(mkprop(agent=pid, conf=cf, v=v, r=r, val=val, s=s, title=title))
        if d.status == ProposalStatus.ACCEPTED:
            n_acc += 1
    print("\n  PASS: " + str(n_acc) + "/3 accepted")


if __name__ == "__main__":
    print("=" * 60)
    print("DISTRIBUTED MULTI-AGENT CONSENSUS SIMULATION")
    print("=" * 60)
    print()
    for a in mkagents(4):
        print(
            "  Agent: " + a.agent_id.ljust(6)
            + " acc=" + str(a.base_accuracy)
            + " bias=" + str(a.bias_toward_accept)
            + " w=" + str(a.initial_weight)
        )
    test_unanimous()
    test_rejection()
    test_mixed()
    test_multiround()
    print()
    print("=" * 60)
    print("ALL SCENARIOS PASSED")
    print("=" * 60)
