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

    def __init__(self, name=None):
        if name is not None:
            self.name = name
        self.tasks = []

    async def run(self, task):
        self.tasks.append(task)

        class Result:
            messages = [
                type(
                    "Message",
                    (),
                    {
                        "content": (
                            '{"veracity":1,"rationality":1,"value":1,'
                            '"security":1,"reasoning":"agent ok"}'
                        )
                    },
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


def test_mem0_memory_tools_add_and_search():
    backend = Mem0MemoryBackend(client=FakeMem0())
    search_memory, add_memory = build_memory_tools(backend, user_id="agent_a")

    import asyncio

    asyncio.run(add_memory("Important TAMAS fact"))
    result = asyncio.run(search_memory("TAMAS"))

    assert "Important TAMAS fact" in result


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
            "content": "A useful result worth remembering.",
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


def test_unified_config_selects_magentic_one_and_consensus():
    config = TAMASRunConfig.from_unified_config(
        "src/mas_framework/configs/experiment_configs/unified_config.yaml"
    )

    assert config.mode == "magentic_one"
    assert config.consensus_enabled is True
    assert config.consensus_strategy == "smart_quorum"
    assert config.honest_model in config.model_capability_coefficients
    assert config.byzantine_model in config.model_capability_coefficients
    assert config.model_capability_coefficients[config.honest_model] == 8.61
    assert config.model_capability_coefficients[config.byzantine_model] == 7.25
