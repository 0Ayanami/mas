from mas_framework.consensus import EventDrivenHotStuffPolicy, SmartQuorumPolicy
from mas_framework.memory import Mem0MemoryBackend
from mas_framework.models import AgentConfig, AgentState, MemoryProposal, ProposalStatus, VerificationVector
from mas_framework.orchestrator import Orchestrator


def make_proposal() -> MemoryProposal:
    return MemoryProposal(
        agent_id="agent_a",
        task_id="task",
        memory_type="research_note",
        title="Consensus memory proposal",
        thoughts_decision="Use proposal verification before memory sync.",
        action="Summarized document",
        data={"evidence": "research note"},
        results_observation="The proposal captures a useful milestone for consensus memory.",
        self_confidence=0.8,
    )


class FakeMem0Client:
    def __init__(self):
        self.added = []
        self.updated = []

    def add(self, messages, **kwargs):
        memory_id = f"mem-{len(self.added) + 1}"
        self.added.append({"id": memory_id, "messages": messages, "kwargs": kwargs})
        return {"results": [{"id": memory_id}]}

    def search(self, query, **kwargs):
        return {
            "results": [
                {
                    "id": item["id"],
                    "memory": item["messages"][0]["content"],
                    "metadata": item["kwargs"].get("metadata", {}),
                }
                for item in self.added
                if query.lower() in item["messages"][0]["content"].lower()
            ]
        }

    def update(self, memory_id, **kwargs):
        self.updated.append({"id": memory_id, "kwargs": kwargs})
        return {"id": memory_id, "event": "UPDATE"}


def test_mem0_backend_add_search_update_proposal():
    client = FakeMem0Client()
    backend = Mem0MemoryBackend(client=client, default_user_id="shared")
    proposal = make_proposal()

    result = backend.add_proposal(proposal, user_id="agent_a")
    assert result["results"][0]["id"] == "mem-1"
    assert client.added[0]["kwargs"]["user_id"] == "agent_a"
    assert client.added[0]["kwargs"]["metadata"]["proposal_id"] == proposal.proposal_id

    hits = backend.search("Consensus", user_id="agent_a")
    assert hits[0]["metadata"]["task_id"] == "task"

    updated = backend.update_proposal("mem-1", proposal, user_id="agent_a")
    assert updated["event"] == "UPDATE"
    assert client.updated[0]["kwargs"]["metadata"]["proposal_id"] == proposal.proposal_id


def test_memory_proposal_dump_uses_restructured_header_body_names():
    proposal = MemoryProposal(
        header={
            "proposal_id": "550e8400-e29b-41d4-a716-446655440000",
            "task_id": "task_20250601_001",
            "timestamp": "2026-06-08T14:30:00.000Z",
            "agent_id": "agent_001",
            "agent_signature": "SHA256_RSA_SIG_BASE64",
            "parent_proposals": ["550e8400-...-0001", "550e8400-...-0002"],
            "body_hash": "will-be-overwritten",
            "proposal_summary": "通过Python requests库获取天气API数据，解析JSON返回温度信息",
        },
        body={
            "thoughts": {
                "thoughts_abstract": "用户需要查询北京今日天气，决定调用公开天气API获取实时数据",
                "key_decisions": [
                    {"decision": "选择OpenWeatherMap作为数据源", "result": "adopted"},
                    {"decision": "使用GET请求而非POST", "result": "adopted"},
                ],
            },
            "actions": [
                {
                    "action_id": "act_1",
                    "type": "api_call",
                    "tool": "http_requests",
                    "params": {
                        "url": "https://api.weather.com/v1/current?city=Beijing",
                        "method": "GET",
                    },
                    "status": "success",
                }
            ],
            "data": [
                {
                    "source": "weather_api",
                    "content_snippet": "北京今日气温25°C，晴，湿度45%",
                    "url": "https://api.weather.com/v1/current?city=Beijing",
                    "timestamp": "2026-06-08T14:30:00.000Z",
                }
            ],
            "observations": [
                {
                    "type": "data_retrieval",
                    "description": "成功获取天气数据并解析JSON",
                    "status": "complete",
                }
            ],
        },
    )

    dumped = proposal.model_dump()
    assert dumped["header"]["agent_id"] == "agent_001"
    assert dumped["header"]["agent_signature"] == "SHA256_RSA_SIG_BASE64"
    assert dumped["header"]["parent_proposals"] == ["550e8400-...-0001", "550e8400-...-0002"]
    assert "proposing_agent_id" not in dumped["header"]
    assert "parent_proposal_ids" not in dumped["header"]
    assert dumped["body"]["thoughts"]["key_decisions"][0]["result"] == "adopted"
    assert dumped["body"]["data"][0]["content_snippet"] == "北京今日气温25°C，晴，湿度45%"


def test_smart_quorum_accepts_confident_majority():
    proposal = make_proposal()
    proposal.verifications = [
        VerificationVector.from_binary_votes(
            veracity=True,
            rationality=True,
            value=True,
            security=True,
            rationale="ok",
            verifier_id="v1",
        ),
        VerificationVector.from_binary_votes(
            veracity=True,
            rationality=True,
            value=True,
            security=True,
            rationale="ok",
            verifier_id="v2",
        ),
        VerificationVector.from_binary_votes(
            veracity=False,
            rationality=True,
            value=True,
            security=True,
            rationale="weak evidence",
            verifier_id="v3",
        ),
        VerificationVector.from_binary_votes(
            veracity=True,
            rationality=True,
            value=True,
            security=True,
            rationale="ok",
            verifier_id="v4",
        ),
    ]

    decision = SmartQuorumPolicy().decide(proposal, agent_count=4)

    assert decision.result == ProposalStatus.ACCEPTED
    assert decision.vote_weight == 4.0


def test_event_driven_hotstuff_forms_three_qcs_and_excludes_proposer_vote():
    proposal = make_proposal()
    proposal.verifications = [
        VerificationVector.from_binary_votes(
            veracity=False,
            rationality=False,
            value=False,
            security=False,
            rationale="self vote should not drive consensus",
            verifier_id="agent_a",
        ),
        VerificationVector.from_binary_votes(
            veracity=True,
            rationality=True,
            value=True,
            security=True,
            rationale="ok",
            verifier_id="v1",
        ),
        VerificationVector.from_binary_votes(
            veracity=True,
            rationality=True,
            value=True,
            security=True,
            rationale="ok",
            verifier_id="v2",
        ),
        VerificationVector.from_binary_votes(
            veracity=False,
            rationality=False,
            value=False,
            security=False,
            rationale="reject",
            verifier_id="v3",
        ),
    ]

    decision = EventDrivenHotStuffPolicy().decide(proposal, agent_count=4)

    assert decision.result == ProposalStatus.ACCEPTED
    assert decision.voting_agents == 3
    assert decision.vote_weight == 2.0
    assert [qc["phase"] for qc in proposal.hotstuff_qcs] == ["prepare", "pre_commit", "commit"]
    assert all(qc["accepted"] for qc in proposal.hotstuff_qcs)
    assert proposal.hotstuff_events[0]["event_type"] == "proposal"
    assert proposal.hotstuff_events[-1]["event_type"] == "decide"


def test_event_driven_hotstuff_rejects_when_prepare_qc_is_missing():
    proposal = make_proposal()
    proposal.verifications = [
        VerificationVector.from_binary_votes(
            veracity=True,
            rationality=True,
            value=True,
            security=True,
            rationale="ok",
            verifier_id="v1",
        ),
        VerificationVector.from_binary_votes(
            veracity=False,
            rationality=False,
            value=False,
            security=False,
            rationale="reject",
            verifier_id="v2",
        ),
        VerificationVector.from_binary_votes(
            veracity=False,
            rationality=False,
            value=False,
            security=False,
            rationale="reject",
            verifier_id="v3",
        ),
    ]

    decision = EventDrivenHotStuffPolicy().decide(proposal, agent_count=4)

    assert decision.result == ProposalStatus.REJECTED
    assert len(proposal.hotstuff_qcs) == 1
    assert proposal.hotstuff_qcs[0]["phase"] == "prepare"
    assert proposal.hotstuff_qcs[0]["accepted"] is False


def test_orchestrator_uses_event_driven_hotstuff_by_default(monkeypatch):
    import mas_framework.orchestrator as orchestrator_module

    monkeypatch.setattr(orchestrator_module, "create_agent", lambda config: None)

    orch = Orchestrator(agent_configs=[AgentConfig(agent_id="agent_a")])

    assert orch.policy.__class__.__name__ == "EventDrivenHotStuffPolicy"


class FakeMemory:
    def __init__(self):
        self.proposals = []

    def add_proposal(self, proposal, user_id):
        self.proposals.append((proposal, user_id))
        return {"ok": True}


class FakeAgent:
    def __init__(self, agent_id, responses):
        self.config = AgentConfig(agent_id=agent_id, role=agent_id, system_prompt="")
        self.state = AgentState()
        self.memory = FakeMemory()
        self._responses = list(responses)

    def run(self, prompt):
        if self._responses:
            return self._responses.pop(0)
        return '{"veracity": true, "rationality": true, "value": true, "security": true, "rationale": "ok"}'


def test_memory_trace_workflow_commits_accepted_proposal_to_all_memories():
    proposer = FakeAgent(
        "researcher_1",
        [
            '{"propose_memory": true, "signal": "MEMORY_PROPOSE", "memory_type": "evidence", "rationale": "useful evidence"}',
            """
            {
              "proposal_summary": "Reusable benchmark evidence was found.",
              "thoughts": {"thoughts_abstract": "The benchmark evidence is relevant.", "key_decisions": []},
              "actions": [{"action_id": "act_1", "type": "tool_call", "tool": "benchmark", "params": {}, "status": "success"}],
              "data": [{"source": "benchmark", "content_snippet": "reusable evidence", "url": "", "timestamp": ""}],
              "observations": [{"type": "task_progress", "description": "evidence can support other agents", "status": "complete"}]
            }
            """,
            '{"veracity": true, "rationality": true, "value": true, "security": true, "rationale": "self ok"}',
        ],
    )
    validator_1 = FakeAgent("method_critic", [])
    validator_2 = FakeAgent("security_verifier", [])
    validator_3 = FakeAgent("systems_verifier", [])

    orch = Orchestrator.__new__(Orchestrator)
    orch.agents = {
        proposer.config.agent_id: proposer,
        validator_1.config.agent_id: validator_1,
        validator_2.config.agent_id: validator_2,
        validator_3.config.agent_id: validator_3,
    }
    orch.agent_count = len(orch.agents)
    orch.policy = SmartQuorumPolicy()

    proposal = orch.propose_memory_from_trace(
        agent_id="researcher_1",
        task_id="agentdojo-task-1",
        react_trace=(
            "Thought: useful finding\n"
            "Action: inspect benchmark\n"
            "Observation: reusable evidence found"
        ),
        task_context="AgentDojo-style benchmark scenario",
    )

    assert proposal is not None
    assert proposal.header.task_id == "agentdojo-task-1"
    assert proposal.header.memory_type == "evidence"
    assert proposal.model_dump()["header"]["agent_id"] == "researcher_1"
    assert "key_decisions" in proposal.model_dump()["body"]["thoughts"]
    assert proposal.verification.self_verification.confidence >= proposer.config.conf_threshold
    assert proposal.consensus_result.result == ProposalStatus.ACCEPTED
    for agent in orch.agents.values():
        assert agent.memory.proposals[0][0].proposal_id == proposal.proposal_id
