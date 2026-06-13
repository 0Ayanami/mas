"""Multi-Agent Collaboration Workflow using CAMEL-AI.

This example demonstrates a complete multi-agent collaboration workflow:
1. Multiple specialized agents work together on a shared task
2. Agents propose findings as structured memory proposals
3. Other agents verify proposals through multi-dimension scoring
4. A smart quorum consensus decides which proposals are accepted
5. Accepted findings are committed to shared memory

Agents:
  - Researcher: Analyzes content and creates proposals
  - Critic: Reviews for methodology and logical soundness
  - Verifier: Checks for security, factuality, and relevance
  - Synthesizer: Aggregates accepted proposals into a final report

Usage:
  python examples/run_multiagent_collab.py                 # mock mode (no API key needed)
  python examples/run_multiagent_collab.py --topic "..."    # real mode with LLM
  python examples/run_multiagent_collab.py --mock           # force mock mode
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

# ── CAMEL-AI imports ────────────────────────────────────────────────────────
from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.toolkits import FunctionTool
from camel.types import ModelPlatformType, ModelType

# ── Framework imports ───────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mas_framework.models import (
    AgentConfig,
    AgentState,
    ConsensusDecision,
    MemoryProposal,
    ProposalStatus,
    VerificationVector,
    SelfVerification,
    MultiAgentVerificationSummary,
)
from mas_framework.consensus import SmartQuorumPolicy


# ═══════════════════════════════════════════════════════════════════════════════
#  MOCK AGENT — used when no LLM API key is available
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MockAgent:
    """A deterministic mock agent for testing the workflow without LLM calls."""

    config: AgentConfig
    state: AgentState = field(default_factory=AgentState)

    def __post_init__(self):
        self.state.weight = 1.0

    def run(self, prompt: str) -> str:
        """Simulate an LLM response based on role."""
        if "researcher" in self.config.agent_id.lower():
            return json.dumps({
                "title": f"Analysis by {self.config.role}",
                "thoughts_abstract": f"Analyzed the input and identified key patterns using {self.config.role} expertise.",
                "actions": ["Read source material", "Extracted key facts", "Cross-referenced findings"],
                "data": {"key_insight_1": "Important finding A", "key_insight_2": "Important finding B"},
                "observations": ["Data supports the main hypothesis", "Several edge cases identified"],
                "self_confidence": 0.85,
            })
        elif "critic" in self.config.agent_id.lower():
            return json.dumps({
                "veracity": True, "rationality": True,
                "value": True, "security": True,
                "confidence": 0.80,
                "rationale": "Methodology is sound and well-documented.",
            })
        elif "verifier" in self.config.agent_id.lower():
            return json.dumps({
                "veracity": True, "rationality": True,
                "value": True, "security": True,
                "confidence": 0.78,
                "rationale": "Findings are factually supported and present low security risk.",
            })
        elif "synthesizer" in self.config.agent_id.lower():
            return json.dumps({
                "summary": "Synthesized report based on verified proposals.",
                "key_conclusions": ["Conclusion 1", "Conclusion 2"],
                "confidence_level": "high",
            })
        return json.dumps({"response": f"Processed by {self.config.role}"})

    def verify(self, proposal: MemoryProposal) -> VerificationVector:
        """Simulate verification by returning a positive vector."""
        return VerificationVector.from_binary_votes(
            veracity=True, rationality=True,
            value=True, security=True,
            rationale=f"Verified by {self.config.agent_id} — all checks passed.",
            verifier_id=self.config.agent_id,
            conf_threshold=self.config.conf_threshold,
            weight=self.state.weight,
        )

    def update_state(
        self, mac: MultiAgentVerificationSummary, status: ProposalStatus
    ) -> None:
        """Update agent metrics after a consensus decision."""
        self.state.proposal_sum += 1
        if status == ProposalStatus.ACCEPTED:
            self.state.proposal_submitted += 1
        self.state.historical_conf = round(
            self.state.proposal_submitted / self.state.proposal_sum, 4
        ) if self.state.proposal_sum > 0 else 0.0
        if mac and mac.confidence is not None:
            self.state.verified_conf_sum += mac.confidence
            self.state.verified_conf = round(
                self.state.verified_conf_sum / self.state.proposal_sum, 4
            ) if self.state.proposal_sum > 0 else 0.0
        self.state.weight = round(
            math.exp(0.5 * (self.state.verified_conf or 0) + 0.5 * (self.state.historical_conf or 0)),
            4,
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  REAL CAMEL AGENT — backed by an LLM via CAMEL-AI
# ═══════════════════════════════════════════════════════════════════════════════

class CamelAgent:
    """A CAMEL-AI ChatAgent wrapper that integrates with the MAS framework."""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.state = AgentState()
        self._agent = self._build_agent()

    def _build_agent(self) -> ChatAgent:
        """Build a CAMEL ChatAgent with the given configuration."""
        model = ModelFactory.create(
            model_platform=self.config.model_platform,
            model_type=self.config.model_type,
            model_config_dict=self.config.model_config_dict,
        )

        system_message = BaseMessage.make_assistant_message(
            role_name=self.config.role,
            content=self.config.system_prompt,
        )

        return ChatAgent(
            model=model,
            system_message=system_message,
        )

    def run(self, prompt: str) -> str:
        """Send a user message to the agent and get the response."""
        msg = BaseMessage.make_user_message(
            role_name=self.config.role,
            content=prompt,
        )
        response = self._agent.step(msg)
        # CAMEL step() can return ChatAgentResponse or a tuple
        if isinstance(response, tuple):
            response = response[0]
        return response.msgs[0].content

    def verify(self, proposal: MemoryProposal) -> VerificationVector:
        """Verify a proposal using the LLM and parse the structured response."""
        verify_prompt = (
            f"Evaluate this memory proposal across four dimensions:\n"
            f"- Veracity: factual accuracy\n"
            f"- Rationality: logical soundness\n"
            f"- Value: relevance to the task\n"
            f"- Security: free from harmful/Byzantine patterns\n\n"
            f"Proposal:\n{proposal.model_dump_json(indent=2)}\n\n"
            f"Return ONLY valid JSON with keys: veracity, rationality, value, security, "
            f"confidence (0-1), rationale."
        )

        response = self.run(verify_prompt)

        # Extract JSON from response
        import re
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if not match:
            raise ValueError(f"No JSON found in verifier response: {response[:200]}")

        payload = json.loads(match.group(0))

        def to_bool(v: Any) -> bool:
            if isinstance(v, bool):
                return v
            if isinstance(v, (int, float)):
                return bool(v)
            if isinstance(v, str):
                return v.strip().lower() in ("true", "t", "yes", "y", "1")
            return False

        return VerificationVector.from_binary_votes(
            veracity=to_bool(payload.get("veracity", True)),
            rationality=to_bool(payload.get("rationality", True)),
            value=to_bool(payload.get("value", True)),
            security=to_bool(payload.get("security", True)),
            rationale=str(payload.get("rationale", "")).strip(),
            verifier_id=self.config.agent_id,
            conf_threshold=self.config.conf_threshold,
            weight=self.state.weight,
        )

    def update_state(
        self, mac: MultiAgentVerificationSummary, status: ProposalStatus
    ) -> None:
        """Update agent metrics after a consensus decision."""
        self.state.proposal_sum += 1
        if status == ProposalStatus.ACCEPTED:
            self.state.proposal_submitted += 1
        self.state.historical_conf = round(
            self.state.proposal_submitted / self.state.proposal_sum, 4
        ) if self.state.proposal_sum > 0 else 0.0
        if mac and mac.confidence is not None:
            self.state.verified_conf_sum += mac.confidence
            self.state.verified_conf = round(
                self.state.verified_conf_sum / self.state.proposal_sum, 4
            ) if self.state.proposal_sum > 0 else 0.0
        self.state.weight = round(
            math.exp(0.5 * (self.state.verified_conf or 0) + 0.5 * (self.state.historical_conf or 0)),
            4,
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  ORCHESTRATOR — manages the multi-agent collaboration workflow
# ═══════════════════════════════════════════════════════════════════════════════

class MultiAgentCollaborationOrchestrator:
    """Orchestrates multi-agent collaboration with verification and consensus."""

    def __init__(
        self,
        agents: dict[str, Any],
        policy: SmartQuorumPolicy | None = None,
    ):
        self.agents = agents
        self.policy = policy or SmartQuorumPolicy()
        self.shared_memory: list[MemoryProposal] = []

    def run_collaboration_cycle(
        self,
        task: str,
        task_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Run a full multi-agent collaboration cycle.

        Steps:
        1. Task assignment: Researcher analyzes the task
        2. Proposal: Researcher creates a structured memory proposal
        3. Self-verification: Researcher validates its own proposal
        4. Multi-agent verification: Critic and Verifier evaluate the proposal
        5. Consensus: Smart quorum decides acceptance/rejection
        6. Memory commit: Accepted proposals are stored
        7. Cycle repeats across different research angles
        8. Results are aggregated
        """
        tid = task_id or f"collab_{int(time.time())}"

        print(f"\n{'='*70}")
        print(f"  TASK: {task}")
        print(f"  Task ID: {tid}")
        print(f"{'='*70}")

        accepted_proposals: list[MemoryProposal] = []
        rounds = 0

        # ── Phase 1: Research and Proposal ──────────────────────────────
        for agent_id in self.agents:
            if "researcher" in agent_id.lower():
                print(f"\n  [Phase 1] {agent_id} is analyzing the task...")
                proposal = self._create_proposal(agent_id, tid, task)
                if proposal is None:
                    continue

                # ── Phase 2: Self-verification ──────────────────────────
                print(f"  [Phase 2] {agent_id} self-verifying...")
                self._self_verify(agent_id, proposal)
                if proposal.status == ProposalStatus.REJECTED:
                    print(f"  ⨯ Self-verification failed for {agent_id}")
                    continue
                print(f"  ✓ Self-verification passed (confidence={proposal.self_confidence:.2f})")

                # ── Phase 3: Multi-agent verification ───────────────────
                print(f"  [Phase 3] Multi-agent verification...")
                self._multi_agent_verify(proposal)

                # ── Phase 4: Consensus ──────────────────────────────────
                print(f"  [Phase 4] Consensus...")
                decision = self.policy.decide(proposal, agent_count=len(self.agents))
                proposal.status = decision.status

                if decision.status == ProposalStatus.ACCEPTED:
                    print(f"  ✓ ACCEPTED (votes={decision.positive_votes:.2f}/{decision.quorum_size}, "
                          f"confidence={decision.average_confidence:.2f})")
                    accepted_proposals.append(proposal)
                    self.shared_memory.append(proposal)
                else:
                    print(f"  ⨯ REJECTED (votes={decision.positive_votes:.2f}/{decision.quorum_size}, "
                          f"confidence={decision.average_confidence:.2f})")

                # ── Phase 5: State update ───────────────────────────────
                self._update_agent_states(agent_id, proposal, decision)
                rounds += 1

        # ── Phase 6: Synthesis ──────────────────────────────────────────
        print(f"\n  [Phase 6] Generating synthesis report...")
        results = self._synthesize_results(task, accepted_proposals)

        print(f"\n{'='*70}")
        print(f"  COLLABORATION COMPLETE")
        print(f"  Rounds: {rounds} | Accepted: {len(accepted_proposals)} proposals")
        print(f"{'='*70}")

        return results

    def _create_proposal(
        self, agent_id: str, task_id: str, task: str
    ) -> MemoryProposal | None:
        """Ask an agent to analyze the task and create a proposal."""
        agent = self.agents[agent_id]

        prompt = (
            f"As a {agent.config.role}, analyze this task and create a structured proposal:\n\n"
            f"Task: {task}\n\n"
            f"Return a JSON object with:\n"
            f'  - "title": short summary\n'
            f'  - "thoughts_abstract": your reasoning process\n'
            f'  - "actions": list of analysis actions\n'
            f'  - "data": dict of key findings\n'
            f'  - "observations": list of observations\n'
            f'  - "self_confidence": float 0.0-1.0\n'
        )

        try:
            response = agent.run(prompt)
        except Exception as e:
            print(f"  ! Error running {agent_id}: {e}")
            return None

        import re
        match = re.search(r"\{.*\}", response, re.DOTALL)
        parsed = {}
        if match:
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        return MemoryProposal(
            agent_id=agent_id,
            task_id=task_id,
            memory_type="research_note",
            title=parsed.get("title", f"Proposal by {agent.config.role}"),
            thoughts_decision=parsed.get("thoughts_abstract", response[:200]),
            action=parsed.get("actions", ["Analysis"]),
            data=parsed.get("data", {}),
            results_observation=parsed.get("observations", ["Completed analysis"]),
            self_confidence=parsed.get("self_confidence", 0.8),
        )

    def _self_verify(self, agent_id: str, proposal: MemoryProposal) -> None:
        """Self-verify a proposal. If it passes, record the self-verification."""
        agent = self.agents[agent_id]
        vector = agent.verify(proposal)
        if vector.vote_result:
            proposal.verifications.append(vector)
            proposal.verification.self_verification = SelfVerification(
                veracity=vector.veracity,
                rationality=vector.rationality,
                value=vector.value,
                security=vector.security,
                confidence=vector.confidence,
                rationale=vector.rationale,
            )
            proposal.status = ProposalStatus.PENDING
        else:
            proposal.status = ProposalStatus.REJECTED

    def _multi_agent_verify(self, proposal: MemoryProposal) -> None:
        """Have all non-proposer agents verify the proposal."""
        proposer = proposal.header.proposing_agent_id
        for agent_id, agent in self.agents.items():
            if agent_id == proposer:
                continue
            try:
                vector = agent.verify(proposal)
                proposal.verifications.append(vector)
                print(f"    {agent_id}: vote={'✓' if vector.vote_result else '⨯'} "
                      f"(v={vector.veracity}, r={vector.rationality}, "
                      f"val={vector.value}, s={vector.security}, "
                      f"c={vector.confidence:.2f}, w={vector.weight:.2f})")
            except Exception as e:
                print(f"    {agent_id}: verification error — {e}")

    def _update_agent_states(
        self, proposer_id: str, proposal: MemoryProposal, decision: ConsensusDecision
    ) -> None:
        """Update all agents' states after a consensus round."""
        mac = proposal.verification.multi_agent_verification
        for agent_id, agent in self.agents.items():
            agent.update_state(mac, decision.status)

    def _synthesize_results(
        self, task: str, proposals: list[MemoryProposal]
    ) -> list[dict[str, Any]]:
        """Aggregate accepted proposals into a structured result."""
        results = []
        for p in proposals:
            results.append({
                "proposal_id": p.header.proposal_id[:8],
                "agent_id": p.header.proposing_agent_id,
                "title": p.title,
                "thoughts": p.thoughts_decision,
                "actions": p.action,
                "data": p.data,
                "observations": p.results_observation,
                "confidence": p.self_confidence,
                "verification_summary": {
                    "veracity": p.verification.multi_agent_verification.veracity,
                    "rationality": p.verification.multi_agent_verification.rationality,
                    "value": p.verification.multi_agent_verification.value,
                    "security": p.verification.multi_agent_verification.security,
                    "overall_confidence": p.verification.multi_agent_verification.confidence,
                },
                "consensus": p.verification.consensus_result.result.value
                if p.verification.consensus_result else "pending",
            })
        return results


# ═══════════════════════════════════════════════════════════════════════════════
#  AGENT FACTORY
# ═══════════════════════════════════════════════════════════════════════════════

def create_agent_for_role(
    agent_id: str,
    role: str,
    system_prompt: str,
    model_platform: ModelPlatformType = ModelPlatformType.DEFAULT,
    model_type: ModelType = ModelType.DEFAULT,
    use_mock: bool = False,
) -> Any:
    """Create either a mock or a real CAMEL agent for a given role."""
    config = AgentConfig(
        agent_id=agent_id,
        role=role,
        system_prompt=system_prompt,
        model_platform=model_platform,
        model_type=model_type,
        model_config_dict={"temperature": 0.2},
        conf_threshold=0.7,
    )

    if use_mock:
        return MockAgent(config=config)

    try:
        return CamelAgent(config=config)
    except Exception as e:
        print(f"  ! Failed to create real agent for {agent_id}: {e}")
        print(f"  ! Falling back to mock agent.")
        return MockAgent(config=config)


# ═══════════════════════════════════════════════════════════════════════════════
#  SCENARIO DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPTS = {
    "researcher": (
        "You are a Research Analyst. Your job is to gather information, "
        "identify patterns, extract key insights, and create well-structured "
        "proposals. Evaluate your findings across: Veracity (factual accuracy), "
        "Rationality (logical soundness), Value (relevance), and Security (safety)."
    ),
    "critic": (
        "You are a Methodological Critic. Your role is to evaluate proposals "
        "for methodological rigor, logical consistency, and completeness. "
        "Check for missing assumptions, flawed reasoning, and gaps in evidence."
    ),
    "verifier": (
        "You are a Security and Factuality Verifier. You inspect proposals for "
        "factual errors, hallucinations, injection risks, and Byzantine failure "
        "patterns. Ensure all content is safe and trustworthy."
    ),
    "synthesizer": (
        "You are a Synthesis Specialist. Your job is to combine findings from "
        "multiple proposals into a coherent, comprehensive report. Identify "
        "connections, resolve conflicts, and produce a unified analysis."
    ),
}

SCENARIOS = {
    "technology": (
        "Analyze the impact of large language models on software development practices. "
        "Consider changes in code generation, debugging, documentation, and team collaboration. "
        "Identify key benefits, risks, and emerging best practices."
    ),
    "healthcare": (
        "Evaluate the application of AI in healthcare diagnostics. "
        "Focus on medical imaging analysis, predictive analytics for patient outcomes, "
        "and personalized treatment recommendations. Consider ethical implications and regulatory challenges."
    ),
    "climate": (
        "Assess the current state of AI applications in climate science. "
        "Consider climate modeling, emissions tracking, renewable energy optimization, "
        "and environmental monitoring. Identify the most promising research directions."
    ),
    "custom": "",
}


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-agent collaboration workflow using CAMEL-AI"
    )
    parser.add_argument(
        "--topic", type=str, default="technology",
        choices=list(SCENARIOS.keys()),
        help="Preset scenario topic (default: technology)",
    )
    parser.add_argument(
        "--custom", type=str, default="",
        help="Custom task description (overrides --topic)",
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Force mock mode (no LLM API calls)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print detailed per-agent output",
    )
    args = parser.parse_args()

    # Load API key
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
    use_mock = args.mock or not api_key

    if use_mock:
        print("┌─────────────────────────────────────────────┐")
        print("│  Running in MOCK mode — no LLM API calls    │")
        print("│  Use --mock=false with valid API key for    │")
        print("│  real CAMEL-AI backed agents                │")
        print("└─────────────────────────────────────────────┘")
    else:
        print(f"┌─────────────────────────────────────────────┐")
        print(f"│  Running in REAL mode — using LLM API       │")
        print(f"└─────────────────────────────────────────────┘")

    # Select task
    task = args.custom if args.custom else SCENARIOS[args.topic]
    print(f"\n  Scenario: {args.topic}")
    print(f"  Task: {task[:80]}...")

    # ── Create agents ───────────────────────────────────────────────────
    print(f"\n  Creating agents...")

    agents = {}
    agent_roles = [
        ("researcher_1", "Researcher", SYSTEM_PROMPTS["researcher"]),
        ("method_critic", "Critic", SYSTEM_PROMPTS["critic"]),
        ("security_verifier", "Verifier", SYSTEM_PROMPTS["verifier"]),
        ("synthesis_agent", "Synthesizer", SYSTEM_PROMPTS["synthesizer"]),
    ]

    for agent_id, role, system_prompt in agent_roles:
        agent = create_agent_for_role(
            agent_id=agent_id,
            role=role,
            system_prompt=system_prompt,
            use_mock=use_mock,
        )
        agents[agent_id] = agent
        print(f"  ✓ Created {agent_id} ({role})")

    # ── Run collaboration ──────────────────────────────────────────────
    orchestrator = MultiAgentCollaborationOrchestrator(agents=agents)

    results = orchestrator.run_collaboration_cycle(task=task)

    # ── Display results ────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*70}")

    for i, r in enumerate(results, 1):
        print(f"\n  [{i}] {r['title']}")
        print(f"      Agent: {r['agent_id']}")
        print(f"      Proposal: {r['proposal_id']}")
        print(f"      Confidence: {r['confidence']:.2f}")
        if r.get("data"):
            print(f"      Key findings: {json.dumps(r['data'], ensure_ascii=False)[:100]}")
        if r.get("verification_summary"):
            vs = r["verification_summary"]
            print(f"      Verification — veracity={vs['veracity']}, "
                  f"rationality={vs['rationality']}, "
                  f"value={vs['value']}, security={vs['security']}")
            print(f"      Overall confidence: {vs['overall_confidence']:.2f}")

    if not results:
        print(f"\n  No proposals were accepted during this cycle.")
        print(f"  This can happen when agent confidence is below the consensus threshold.")
        print(f"  Tip: In mock mode, all verifications pass automatically. "
              f"Run with --topic to try different scenarios.")

    # Save results
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"collab_results_{int(time.time())}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "task": task,
            "topic": args.topic,
            "mode": "mock" if use_mock else "real",
            "num_accepted": len(results),
            "results": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
