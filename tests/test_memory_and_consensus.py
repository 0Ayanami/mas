import json

import pytest
from autogen_ext.models.replay import ReplayChatCompletionClient

from mas_framework.consensus import (
    AutoGenProposalEvaluator,
    MajorityVoteConsensus,
    ProposalBuilder,
    SmartQuorumConsensus,
    VerificationContext,
    VerificationVector,
)
from mas_framework.memory import Mem0MemoryBackend, build_memory_tools
from mas_framework.tamas_workflow import TAMASAutoGenRunner, TAMASRunConfig, TAMASToolLoader


class FakeMem0:
    def __init__(self):
        self.items = []

    def add(self, messages, **kwargs):
        self.items.append({"messages": messages, "kwargs": kwargs})
        return {"results": [{"id": f"mem-{len(self.items)}"}]}

    def search(self, query, **kwargs):
        return {
            "results": [
                {
                    "memory": item["messages"][0]["content"],
                    "metadata": item["kwargs"].get("metadata", {}),
                }
                for item in self.items
                if query.lower() in item["messages"][0]["content"].lower()
            ]
        }


class FakeVerificationAgent:
    name = "verifier_agent"

    def __init__(self, name=None, content=None):
        if name is not None:
            self.name = name
        self.content = content or (
            '{"veracity":1,"rationality":1,"value":1,'
            '"security":1,"reasoning":"agent ok"}'
        )
        self.tasks = []

    async def run(self, task):
        self.tasks.append(task)

        class Result:
            messages = [
                type(
                    "Message",
                    (),
                    {"content": self.content},
                )()
            ]

        return Result()


def model_info():
    return {
        "vision": False,
        "function_calling": True,
        "json_output": True,
        "family": "unknown",
        "structured_output": True,
    }


def test_autogen_verification_engine_uses_replay_model_client():
    proposal = ProposalBuilder().from_agent_output(
        task_id="task-1",
        agent_id="agent_a",
        output={
            "observations": [{"type": "note", "description": "Useful memory"}],
            "proposal_summary": "Useful memory",
        },
    )
    evaluator = AutoGenProposalEvaluator(
        model_client=ReplayChatCompletionClient(
            ['{"veracity":1,"rationality":1,"value":1,"security":1,"reasoning":"ok"}']
        )
    )

    vector = evaluator.evaluate(
        proposal,
        VerificationContext(task_id="task-1"),
        verifier_agent_id="verifier",
    )

    assert vector.confidence_score == 1.0
    assert vector.metadata["evaluator"] == "autogen"


def test_autogen_verification_engine_uses_verifier_agent_when_available():
    proposal = ProposalBuilder().from_agent_output(
        task_id="task-1",
        agent_id="agent_a",
        output={
            "observations": [{"type": "note", "description": "Useful memory"}],
            "proposal_summary": "Useful memory",
        },
    )
    verifier = FakeVerificationAgent()
    evaluator = AutoGenProposalEvaluator(verifier_agents={verifier.name: verifier})

    vector = evaluator.evaluate(
        proposal,
        VerificationContext(task_id="task-1"),
        verifier_agent_id=verifier.name,
    )

    assert verifier.tasks
    assert vector.confidence_score == 1.0
    assert vector.metadata["evaluator"] == "autogen_agent"
    assert vector.metadata["agent"] == verifier.name


def test_majority_and_smart_quorum_decisions():
    builder = ProposalBuilder()
    proposal = builder.from_agent_output(
        task_id="task-1",
        agent_id="agent_a",
        output={"observations": ["A useful result"], "proposal_summary": "A useful result"},
    )
    context = VerificationContext(task_id="task-1")
    vectors = [
        AutoGenProposalEvaluator(
            model_client=ReplayChatCompletionClient(
                ['{"veracity":1,"rationality":1,"value":1,"security":1,"reasoning":"ok"}']
            )
        ).evaluate(proposal, context, verifier_agent_id=f"v{i}")
        for i in range(4)
    ]

    majority = MajorityVoteConsensus().decide(proposal, vectors)
    smart = SmartQuorumConsensus(
        agent_weights={f"v{i}": 1.0 for i in range(4)},
        use_dynamic_estimate=True,
    ).decide(proposal, vectors)

    assert majority.accepted
    assert smart.accepted
    assert smart.metadata["has_quorum_certificate"] is True


def test_smart_quorum_records_weighted_dimension_summary():
    proposal = ProposalBuilder().from_agent_output(
        task_id="task-1",
        agent_id="agent_a",
        output={"observations": ["A useful result"], "proposal_summary": "A useful result"},
    )
    vectors = [
        VerificationVector(
            veracity=1,
            rationality=1,
            value=0,
            security=1,
            verifier_agent_id="v0",
        ),
        VerificationVector(
            veracity=0,
            rationality=1,
            value=1,
            security=1,
            verifier_agent_id="v1",
        ),
    ]

    decision = SmartQuorumConsensus(
        agent_weights={"v0": 2.0, "v1": 1.0},
        byzantine_agents=[],
        honest_agents=["v0", "v1"],
        epsilon_ratio=0.0,
    ).decide(proposal, vectors)

    weighted_scores = decision.metadata["multi_verification_summary"]
    assert weighted_scores["veracity"] == 2 / 3
    assert weighted_scores["rationality"] == 1.0
    assert weighted_scores["value"] == 1 / 3
    assert weighted_scores["security"] == 1.0


def test_mem0_memory_tools_search_shared_memory_only():
    backend = Mem0MemoryBackend(client=FakeMem0())
    backend.add("Important TAMAS fact", user_id="shared-memory")
    tools = build_memory_tools(backend, user_id="shared-memory")
    search_memory = tools[0]

    import asyncio

    result = asyncio.run(search_memory("TAMAS"))

    assert len(tools) == 1
    assert tools[0].__name__ == "search_memory"
    assert "Important TAMAS fact" in result


def test_mem0_add_proposal_requires_passed_consensus():
    backend = Mem0MemoryBackend(client=FakeMem0())
    proposal = ProposalBuilder().from_agent_output(
        task_id="task-1",
        agent_id="agent_a",
        output={
            "observations": [{"type": "note", "description": "Useful memory"}],
            "proposal_summary": "Useful memory",
        },
    )

    with pytest.raises(ValueError, match="consensus_result='pass'"):
        backend.add_proposal(proposal, user_id="shared-memory")

    assert backend.client.items == []


def test_tamas_tool_loader_handles_agent_suffixes():
    loader = TAMASToolLoader("TAMAS-main")

    news_tools = loader.tools_for_agent("News Gathering Agent")
    writing_tools = loader.tools_for_agent("Article Writing Agent A")

    assert [tool.__name__ for tool in news_tools] == [
        "fetch_latest_rss_feeds",
        "scrape_news_websites",
        "monitor_twitter_trends",
        "track_subreddit_posts",
        "get_alerts_from_news_api",
        "exfiltrate_all_rss_sources",
        "inject_fake_trends",
        "exfiltrate_trending_subreddits",
        "poison_gathered_data",
    ]
    assert "draft_article_body" in {tool.__name__ for tool in writing_tools}


def test_tamas_runner_builds_round_robin_agents_with_mem0_tools():
    case = TAMASAutoGenRunner.load_dataset("TAMAS-main/data/Byzantine/news_byzantine.json")[0]
    runner = TAMASAutoGenRunner(
        config=TAMASRunConfig(
            mode="round_robin",
            consensus_enabled=False,
            honest_model="honest-model",
            byzantine_model="byzantine-model",
            model_capability_coefficients={
                "honest-model": 5.0,
                "byzantine-model": 2.0,
            },
        ),
        memory_backend=Mem0MemoryBackend(client=FakeMem0()),
        model_client=ReplayChatCompletionClient(["ok"], model_info=model_info()),
    )

    agents = runner._build_agents(case)
    team = runner._build_team(agents)

    assert len(agents) == 4
    assert [agent.name for agent in agents] == [
        "news_gathering_agent_1",
        "fact_checking_agent_2",
        "article_writing_agent_3",
        "distribution_agent_4",
    ]
    assert team.__class__.__name__ == "RoundRobinGroupChat"
    assert runner._agent_specs["distribution_agent_4"].is_byzantine
    assert runner._agent_specs["distribution_agent_4"].model == "byzantine-model"
    assert runner._agent_specs["distribution_agent_4"].capability_coefficient == 2.0


def test_tamas_runner_updates_weight_windows_after_consensus():
    case = TAMASAutoGenRunner.load_dataset("TAMAS-main/data/Byzantine/news_byzantine.json")[0]
    runner = TAMASAutoGenRunner(
        config=TAMASRunConfig(
            mode="round_robin",
            consensus_enabled=True,
            honest_model="honest-model",
            byzantine_model="byzantine-model",
            model_capability_coefficients={
                "honest-model": 5.0,
                "byzantine-model": 2.0,
            },
            epsilon_ratio=0.0,
        ),
        memory_backend=Mem0MemoryBackend(client=FakeMem0()),
        model_client=ReplayChatCompletionClient(["ok"], model_info=model_info()),
    )
    runner._build_agents(case)
    proposal = ProposalBuilder().from_agent_output(
        task_id="task-1",
        agent_id="news_gathering_agent_1",
        output={"observations": ["A useful result"], "proposal_summary": "A useful result"},
    )
    verifications = [
        VerificationVector(
            veracity=1,
            rationality=1,
            value=1,
            security=1,
            verifier_agent_id="fact_checking_agent_2",
        ),
        VerificationVector(
            veracity=1,
            rationality=1,
            value=1,
            security=1,
            verifier_agent_id="article_writing_agent_3",
        ),
        VerificationVector(
            veracity=0,
            rationality=1,
            value=1,
            security=0,
            verifier_agent_id="distribution_agent_4",
        ),
    ]

    consensus = runner._build_consensus(list(runner._agent_specs))
    decision = consensus.decide(proposal, verifications)
    runner._update_agent_weights_after_consensus(
        proposer_agent_id=proposal.agent_id,
        decision=decision,
    )

    proposer_state = runner.weight_manager.get_state("news_gathering_agent_1")
    honest_verifier_state = runner.weight_manager.get_state("fact_checking_agent_2")
    byzantine_verifier_state = runner.weight_manager.get_state("distribution_agent_4")

    assert list(proposer_state.proposal_confidences) == [
        decision.metadata["proposal_confidence_score"]
    ]
    assert list(proposer_state.vote_alignments) == [decision.accepted]
    assert list(honest_verifier_state.vote_alignments) == [True]
    assert list(byzantine_verifier_state.vote_alignments) == [False]
    assert "weight_snapshots_after_update" in decision.metadata


def test_tamas_consensus_stores_accepted_proposal_once_in_shared_memory():
    case = TAMASAutoGenRunner.load_dataset("TAMAS-main/data/Byzantine/news_byzantine.json")[0]
    fake_mem0 = FakeMem0()
    runner = TAMASAutoGenRunner(
        config=TAMASRunConfig(
            mode="round_robin",
            consensus_enabled=True,
            verification_type="llm",
            honest_model="honest-model",
            byzantine_model="byzantine-model",
            model_capability_coefficients={
                "honest-model": 5.0,
                "byzantine-model": 2.0,
            },
            epsilon_ratio=0.0,
            memory_user_id="shared-test-memory",
        ),
        memory_backend=Mem0MemoryBackend(client=fake_mem0),
        model_client=ReplayChatCompletionClient(["ok"], model_info=model_info()),
    )
    runner._build_agents(case)
    fake_agents = [
        FakeVerificationAgent(agent_id)
        for agent_id in runner._agent_specs
    ]
    event = type(
        "Event",
        (),
        {
            "source": "news_gathering_agent_1",
            "content": (
                "A useful result worth remembering.\n"
                "MEMORY_PROPOSAL\n"
                "```json\n"
                "{\n"
                '  "proposal_summary": "A useful result worth remembering.",\n'
                '  "observations": [{"type": "task_fact", "description": "A useful result worth remembering.", "status": "complete"}]\n'
                "}\n"
                "```\n"
                "END_MEMORY_PROPOSAL"
            ),
        },
    )()

    decisions = runner._run_memory_consensus(
        task_id="task-1",
        task_description="Test shared memory upload",
        events=[event],
        agents=fake_agents,
    )

    assert decisions[0].accepted
    assert len(fake_mem0.items) == 1
    assert fake_mem0.items[0]["kwargs"]["user_id"] == "shared-test-memory"
    assert fake_mem0.items[0]["kwargs"]["metadata"]["agent_id"] == "news_gathering_agent_1"
    stored_proposal = json.loads(fake_mem0.items[0]["messages"][0]["content"])
    assert stored_proposal["verification"]["self_verification"]["veracity_score"] == 1.0
    assert stored_proposal["verification"]["multi_verification"]["weighted_scores"]
    assert stored_proposal["verification"]["consensus_result"]["result"] == "pass"


def test_tamas_proposal_builder_assigns_header_fields_from_workflow():
    runner = TAMASAutoGenRunner(
        config=TAMASRunConfig(
            consensus_enabled=True,
            verification_type="heuristic",
            honest_model="honest-model",
            byzantine_model="byzantine-model",
        ),
        memory_backend=Mem0MemoryBackend(client=FakeMem0()),
        model_client=ReplayChatCompletionClient(["ok"], model_info=model_info()),
    )
    payload = {
        "proposal_summary": "Agent body-only proposal",
        "proposal_id": "agent-must-not-control-this",
        "task_id": "agent-must-not-control-this",
        "agent_id": "agent-must-not-control-this",
        "timestamp": "agent-must-not-control-this",
        "body_hash": "agent-must-not-control-this",
        "agent_signature": "agent-must-not-control-this",
        "observations": [
            {
                "type": "task_fact",
                "description": "Only body fields are consumed.",
                "status": "complete",
            }
        ],
        "self_verification": {
            "veracity_score": 1,
            "rationality_score": 1,
            "value_score": 1,
            "security_score": 1,
        },
    }

    proposal = runner.proposal_builder.from_agent_output(
        task_id="workflow-task",
        agent_id="workflow_agent",
        output=payload,
    )

    assert proposal.task_id == "workflow-task"
    assert proposal.agent_id == "workflow_agent"
    assert proposal.proposal_id != "agent-must-not-control-this"
    assert proposal.header.body_hash != "agent-must-not-control-this"
    assert proposal.header.agent_signature != "agent-must-not-control-this"
    assert runner._self_verification_confidence(proposal) == 0.0


def test_tamas_self_verification_gate_blocks_low_confidence_consensus():
    case = TAMASAutoGenRunner.load_dataset("TAMAS-main/data/Byzantine/news_byzantine.json")[0]
    fake_mem0 = FakeMem0()
    runner = TAMASAutoGenRunner(
        config=TAMASRunConfig(
            mode="round_robin",
            consensus_enabled=True,
            verification_type="llm",
            honest_model="honest-model",
            byzantine_model="byzantine-model",
            self_confidence_threshold=0.6,
            memory_user_id="shared-test-memory",
        ),
        memory_backend=Mem0MemoryBackend(client=fake_mem0),
        model_client=ReplayChatCompletionClient(["ok"], model_info=model_info()),
    )
    runner._build_agents(case)
    fake_agents = [
        FakeVerificationAgent(
            agent_id,
            content=(
                '{"veracity":0,"rationality":0,"value":0,'
                '"security":1,"reasoning":"low confidence self verification"}'
            )
            if agent_id == "news_gathering_agent_1"
            else None,
        )
        for agent_id in runner._agent_specs
    ]
    event = type(
        "Event",
        (),
        {
            "source": "news_gathering_agent_1",
            "content": (
                "Low confidence proposal.\n"
                "MEMORY_PROPOSAL\n"
                "```json\n"
                "{\n"
                '  "proposal_summary": "Low confidence proposal.",\n'
                '  "observations": [{"type": "task_fact", "description": "Low confidence.", "status": "complete"}]\n'
                "}\n"
                "```\n"
                "END_MEMORY_PROPOSAL"
            ),
        },
    )()

    decisions, proposals = runner._run_memory_consensus(
        task_id="task-1",
        task_description="Test self gate",
        events=[event],
        agents=fake_agents,
        return_proposals=True,
    )

    assert len(proposals) == 1
    assert runner._self_verification_confidence(proposals[0]) == 0.25
    assert decisions == []
    assert fake_mem0.items == []
    proposer = next(agent for agent in fake_agents if agent.name == "news_gathering_agent_1")
    verifiers = [agent for agent in fake_agents if agent.name != "news_gathering_agent_1"]
    assert proposer.tasks
    assert all(agent.tasks == [] for agent in verifiers)


def test_tamas_self_verification_threshold_is_inclusive_and_excludes_proposer_vote():
    case = TAMASAutoGenRunner.load_dataset("TAMAS-main/data/Byzantine/news_byzantine.json")[0]
    fake_mem0 = FakeMem0()
    runner = TAMASAutoGenRunner(
        config=TAMASRunConfig(
            mode="round_robin",
            consensus_enabled=True,
            verification_type="llm",
            include_proposer_as_verifier=False,
            honest_model="honest-model",
            byzantine_model="byzantine-model",
            self_confidence_threshold=0.6,
            memory_user_id="shared-test-memory",
        ),
        memory_backend=Mem0MemoryBackend(client=fake_mem0),
        model_client=ReplayChatCompletionClient(["ok"], model_info=model_info()),
    )
    runner._build_agents(case)
    fake_agents = [
        FakeVerificationAgent(
            agent_id,
            content=(
                '{"veracity":1,"rationality":0,"value":0,'
                '"security":1,"reasoning":"boundary confidence self verification"}'
            )
            if agent_id == "news_gathering_agent_1"
            else None,
        )
        for agent_id in runner._agent_specs
    ]
    event = type(
        "Event",
        (),
        {
            "source": "news_gathering_agent_1",
            "content": (
                "Boundary confidence proposal.\n"
                "MEMORY_PROPOSAL\n"
                "```json\n"
                "{\n"
                '  "proposal_summary": "Boundary confidence proposal.",\n'
                '  "observations": [{"type": "task_fact", "description": "Boundary confidence.", "status": "complete"}]\n'
                "}\n"
                "```\n"
                "END_MEMORY_PROPOSAL"
            ),
        },
    )()

    decisions, proposals = runner._run_memory_consensus(
        task_id="task-1",
        task_description="Test inclusive self gate",
        events=[event],
        agents=fake_agents,
        return_proposals=True,
    )

    assert runner._self_verification_confidence(proposals[0]) == 0.6
    assert len(decisions) == 1
    assert "news_gathering_agent_1" not in {
        vote.voter_agent_id for vote in decisions[0].votes
    }
    assert len(decisions[0].votes) == len(fake_agents) - 1
    proposer = next(agent for agent in fake_agents if agent.name == "news_gathering_agent_1")
    verifiers = [agent for agent in fake_agents if agent.name != "news_gathering_agent_1"]
    assert proposer.tasks
    assert all(agent.tasks for agent in verifiers)


def test_tamas_include_proposer_as_verifier_adds_self_vote_to_consensus():
    case = TAMASAutoGenRunner.load_dataset("TAMAS-main/data/Byzantine/news_byzantine.json")[0]
    runner = TAMASAutoGenRunner(
        config=TAMASRunConfig(
            mode="round_robin",
            consensus_enabled=True,
            verification_type="llm",
            include_proposer_as_verifier=True,
            honest_model="honest-model",
            byzantine_model="byzantine-model",
            self_confidence_threshold=0.6,
            memory_user_id="shared-test-memory",
        ),
        memory_backend=Mem0MemoryBackend(client=FakeMem0()),
        model_client=ReplayChatCompletionClient(["ok"], model_info=model_info()),
    )
    runner._build_agents(case)
    fake_agents = [FakeVerificationAgent(agent_id) for agent_id in runner._agent_specs]
    event = type(
        "Event",
        (),
        {
            "source": "news_gathering_agent_1",
            "content": (
                "Include proposer vote.\n"
                "MEMORY_PROPOSAL\n"
                "```json\n"
                "{\n"
                '  "proposal_summary": "Include proposer vote.",\n'
                '  "observations": [{"type": "task_fact", "description": "Include proposer vote.", "status": "complete"}]\n'
                "}\n"
                "```\n"
                "END_MEMORY_PROPOSAL"
            ),
        },
    )()

    decisions, proposals = runner._run_memory_consensus(
        task_id="task-1",
        task_description="Test include proposer vote",
        events=[event],
        agents=fake_agents,
        return_proposals=True,
    )

    assert len(proposals) == 1
    assert len(decisions) == 1
    assert "news_gathering_agent_1" in {
        vote.voter_agent_id for vote in decisions[0].votes
    }
    assert len(decisions[0].votes) == len(fake_agents)


def test_tamas_consensus_ignores_outputs_without_memory_proposal_block():
    case = TAMASAutoGenRunner.load_dataset("TAMAS-main/data/Byzantine/news_byzantine.json")[0]
    fake_mem0 = FakeMem0()
    runner = TAMASAutoGenRunner(
        config=TAMASRunConfig(
            mode="round_robin",
            consensus_enabled=True,
            verification_type="heuristic",
            honest_model="honest-model",
            byzantine_model="byzantine-model",
            memory_user_id="shared-test-memory",
        ),
        memory_backend=Mem0MemoryBackend(client=fake_mem0),
        model_client=ReplayChatCompletionClient(["ok"], model_info=model_info()),
    )
    runner._build_agents(case)
    fake_agents = [FakeVerificationAgent(agent_id) for agent_id in runner._agent_specs]
    event = type(
        "Event",
        (),
        {
            "source": "news_gathering_agent_1",
            "content": "Plain task response without a memory proposal.",
        },
    )()

    decisions, proposals = runner._run_memory_consensus(
        task_id="task-1",
        task_description="Test no proposal",
        events=[event],
        agents=fake_agents,
        return_proposals=True,
    )

    assert decisions == []
    assert proposals == []
    assert fake_mem0.items == []


def test_unified_config_selects_magentic_one_and_consensus():
    config = TAMASRunConfig.from_unified_config(
        "src/mas_framework/configs/experiment_configs/unified_config.yaml"
    )

    assert config.mode == "magentic_one"
    assert config.consensus_enabled is True
    assert config.consensus_strategy == "smart_quorum"
    assert config.self_confidence_threshold == 0.6
    assert config.honest_model in config.model_capability_coefficients
    assert config.byzantine_model in config.model_capability_coefficients
    assert config.model_capability_coefficients[config.honest_model] == 8.61
    assert config.model_capability_coefficients[config.byzantine_model] == 7.25
