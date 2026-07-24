import json

from autogen_ext.models.replay import ReplayChatCompletionClient

from mas_framework.consensus import (
    AutoGenProposalEvaluator,
    MajorityVoteConsensus,
    ProposalBuilder,
    SmartQuorumConsensus,
    VerificationContext,
    VerificationVector,
)
from mas_framework.exp_1.models import TaskMemoryPool
from mas_framework.exp_3.models import TAMASRunConfig, TamasGroupSpec
from mas_framework.exp_3.tamas_data import infer_attack_type
from mas_framework.exp_3.tamas_workflow import (
    MEMORY_PROPOSAL_TRIGGER_PROMPT,
    TAMASAutoGenRunner,
    TAMASToolLoader,
)


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
        self.state = {"history": ["before"]}

    async def save_state(self):
        return dict(self.state)

    async def load_state(self, state):
        self.state = dict(state)

    async def run(self, task):
        self.tasks.append(task)
        self.state["history"].append("verification")

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


def agent_tool_names(agent):
    return {getattr(tool, "name", "") for tool in getattr(agent, "_tools", [])}


def tamas_config(method="swarm", **kwargs):
    return TAMASRunConfig(
        config_path="test",
        group_spec=TamasGroupSpec(method=method, n=3, f=1, model_regime="same"),
        **kwargs,
    )


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


def test_autogen_verification_engine_uses_verifier_agent_and_restores_state():
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
    assert verifier.state == {"history": ["before"]}
    assert vector.confidence_score == 1.0
    assert vector.metadata["evaluator"] == "autogen_agent"
    assert vector.metadata["agent"] == verifier.name


def test_majority_and_smart_quorum_decisions():
    builder = ProposalBuilder()
    proposal = builder.from_agent_output(
        task_id="task-1",
        agent_id="agent_a",
        output={
            "observations": [{"type": "note", "description": "A useful result"}],
            "proposal_summary": "A useful result",
        },
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
        output={
            "observations": [{"type": "note", "description": "A useful result"}],
            "proposal_summary": "A useful result",
        },
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


def test_task_memory_pool_is_task_scoped_shared_context():
    proposal = ProposalBuilder().from_agent_output(
        task_id="task-1",
        agent_id="agent_a",
        output={
            "observations": [{"type": "note", "description": "Useful TAMAS fact"}],
            "proposal_summary": "Useful TAMAS fact",
        },
    )
    pool = TaskMemoryPool(task_id="task-1")

    pool.add(proposal)

    message = pool.coordinator_message()
    assert len(pool.payloads()) == 1
    assert "SHARED_TASK_MEMORY" in message
    assert proposal.proposal_id in message
    pool.accepted_proposals.clear()
    assert pool.payloads() == []


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


def test_tamas_workflow_reads_byzantine_agent_type_from_dataset():
    path = "TAMAS-main/data/Byzantine/education_byzantine.json"
    case = TAMASAutoGenRunner.load_dataset(path)[0]
    runner = TAMASAutoGenRunner(
        config=tamas_config(
            honest_model="honest-model",
            byzantine_model="byzantine-model",
        ),
        model_client=ReplayChatCompletionClient(["ok"], model_info=model_info()),
    )
    runner._build_agents(case)

    assert infer_attack_type(path) == "byzantine"
    assert case["_tamas_metadata"]["attack_type"] == "byzantine"
    assert {agent["agent_type"] for agent in case["agents"]} == {"honest", "byzantine"}
    byzantine_specs = [
        spec for spec in runner._agent_specs.values() if spec.is_byzantine
    ]
    honest_specs = [
        spec for spec in runner._agent_specs.values() if not spec.is_byzantine
    ]
    assert [spec.display_name for spec in byzantine_specs] == ["Assessment Agent"]
    assert len(honest_specs) == 3


def test_tamas_workflow_reads_colluding_agent_type_from_dataset():
    path = "TAMAS-main/data/Colluding/education_colluding.json"
    case = TAMASAutoGenRunner.load_dataset(path)[0]
    runner = TAMASAutoGenRunner(
        config=tamas_config(
            honest_model="honest-model",
            byzantine_model="byzantine-model",
        ),
        model_client=ReplayChatCompletionClient(["ok"], model_info=model_info()),
    )
    runner._build_agents(case)

    assert infer_attack_type(path) == "colluding"
    assert case["_tamas_metadata"]["attack_type"] == "colluding"
    assert {agent["agent_type"] for agent in case["agents"]} == {"honest", "byzantine"}
    assert [
        spec.display_name
        for spec in runner._agent_specs.values()
        if spec.is_byzantine
    ] == [
        "Adaptive Learning Agent",
        "Progress Tracking Agent",
    ]


def test_tamas_workflow_reads_contradicting_agent_type_from_dataset():
    path = "TAMAS-main/data/Contradicting/legal_contradicting.json"
    case = TAMASAutoGenRunner.load_dataset(path)[0]
    runner = TAMASAutoGenRunner(
        config=tamas_config(
            honest_model="honest-model",
            byzantine_model="byzantine-model",
        ),
        model_client=ReplayChatCompletionClient(["ok"], model_info=model_info()),
    )
    runner._build_agents(case)

    assert infer_attack_type(path) == "contradicting"
    assert case["_tamas_metadata"]["attack_type"] == "contradicting"
    assert {agent["agent_type"] for agent in case["agents"]} == {"honest", "byzantine"}
    assert [
        spec.display_name
        for spec in runner._agent_specs.values()
        if spec.is_byzantine
    ] == [
        "Document Drafting Agent A",
        "Document Drafting Agent B",
    ]


def test_tamas_runner_builds_swarm_agents_without_memory_when_consensus_disabled():
    case = TAMASAutoGenRunner.load_dataset("TAMAS-main/data/Byzantine/news_byzantine.json")[0]
    runner = TAMASAutoGenRunner(
        config=tamas_config(
            method="swarm",
            honest_model="honest-model",
            byzantine_model="byzantine-model",
            capability_coefficients={
                "honest-model": 5.0,
                "byzantine-model": 2.0,
            },
        ),
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
    assert team.__class__.__name__ == "Swarm"
    assert all(
        {handoff.target for handoff in getattr(agent, "_handoffs", {}).values()}
        == {other.name for other in agents if other.name != agent.name}
        for agent in agents
    )
    assert runner._agent_specs["distribution_agent_4"].is_byzantine
    assert runner._agent_specs["distribution_agent_4"].model == "byzantine-model"
    assert runner._agent_specs["distribution_agent_4"].capability_coefficient == 2.0
    assert all("search_memory" not in agent_tool_names(agent) for agent in agents)
    assert all(
        MEMORY_PROPOSAL_TRIGGER_PROMPT not in getattr(agent, "_system_messages", [""])[0].content
        for agent in agents
    )


def test_tamas_runner_adds_memory_trigger_prompt_when_consensus_enabled():
    case = TAMASAutoGenRunner.load_dataset("TAMAS-main/data/Byzantine/news_byzantine.json")[0]
    runner = TAMASAutoGenRunner(
        config=tamas_config(
            method="swarm_consensus",
            honest_model="honest-model",
            byzantine_model="byzantine-model",
        ),
        model_client=ReplayChatCompletionClient(["ok"], model_info=model_info()),
    )

    agents = runner._build_agents(case)

    assert all(
        MEMORY_PROPOSAL_TRIGGER_PROMPT in getattr(agent, "_system_messages", [""])[0].content
        for agent in agents
    )


def test_exp3_config_loads_single_swarm_consensus_group():
    config = TAMASRunConfig.from_yaml(
        "src/mas_framework/configs/experiment_configs/exp3_tamas.yaml"
    )

    assert config.group_spec.group_id == "swarm_consensus__n3__f1__same"
    assert config.consensus_enabled is True
    assert config.consensus_strategy == "smart_quorum"
    assert config.max_memory_proposals_per_agent_per_case == 3
    assert config.honest_model in config.model_capability_coefficients
    assert config.byzantine_model in config.model_capability_coefficients
    assert config.model_capability_coefficients[config.honest_model] == config.capability_coefficient
    assert json.dumps(config.group_spec.to_dict())
