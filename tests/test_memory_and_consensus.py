from mas_framework.consensus import SmartQuorumPolicy
from mas_framework.memory import SQLiteMemoryStore
from mas_framework.models import MemoryProposal, ProposalStatus, VerificationVector


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


def test_memory_store_roundtrip(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite", tmp_path / "audit.jsonl")
    proposal = make_proposal()
    proposal.status = ProposalStatus.ACCEPTED

    store.save_proposal(proposal)

    loaded = store.get_proposal(proposal.proposal_id)
    assert loaded.proposal_id == proposal.proposal_id
    assert loaded.status == ProposalStatus.ACCEPTED
    assert store.search("Consensus")[0].proposal_id == proposal.proposal_id


def test_smart_quorum_accepts_confident_majority():
    proposal = make_proposal()
    proposal.verifications = [
        VerificationVector.from_binary_votes(
            truthfulness=True,
            rationality=True,
            value=True,
            non_malicious=True,
            rationale="ok",
            verifier_id="v1",
        ),
        VerificationVector.from_binary_votes(
            truthfulness=True,
            rationality=True,
            value=True,
            non_malicious=True,
            rationale="ok",
            verifier_id="v2",
        ),
        VerificationVector.from_binary_votes(
            truthfulness=False,
            rationality=True,
            value=True,
            non_malicious=True,
            rationale="weak evidence",
            verifier_id="v3",
        ),
    ]

    decision = SmartQuorumPolicy().decide(proposal, validator_count=3)

    assert decision.status == ProposalStatus.ACCEPTED
    assert decision.positive_votes == 3

