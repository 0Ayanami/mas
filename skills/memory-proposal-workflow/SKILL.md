# Memory Proposal Workflow Skill

## 概览

Agent 在完成一轮完整的 ReAct 循环（Observation -> Thought -> Action -> Observation）后，若有值得共享的记忆沉淀（research_note / evidence / milestone / tool_observation），应当通过本地创建 **Memory Proposal** 并提交至 Orchestrator 进行多智能体验证与共识决策。

## 工作流

### 1. 何时提出 Proposal

在每轮 ReAct 循环结束时，判断以下任意条件：

- **research_note**：得出新的推论、假设或总结性见解
- **evidence**：收集到对当前任务有支持作用的事实数据
- **milestone**：完成了一个可独立提交的子目标
- **tool_observation**：使用外部工具后获得有价值的观察

### 2. 如何构建 Proposal（本地阶段）

使用 `prepare_proposal_for_submission` 工具（已注册到 ToolRegistry），该工具内部按顺序完成三个步骤：

1. **填充 Header**：ProposalHeader（proposal_id、timestamp、proposing_agent_id、body_hash）
2. **填充 Body**：ProposalBody（thoughts、actions、data、observations）
3. **Self-Verification**：对四个维度（veracity、rationality、value、security）进行自我评分

### 3. 提交验证（共识阶段）

将返回的 proposal JSON 传给 Orchestrator 的 `verify_and_commit` 方法：

- 其他 agent 作为验证者对 proposal 进行多维验证，生成 VerificationVector
- SmartQuorumPolicy 收集验证结果，计算加权赞成率
- 若赞成率 >= 阈值，提案被 ACCEPTED 并持久化到共享记忆
- 否则被 REJECTED

### 4. 完整示例

```python
# 第1步：调用工具构建提案（含自验证）
proposal_json = tools.call(
    "prepare_proposal_for_submission",
    agent_id="researcher_1",
    task_id="task_001",
    current_thoughts="分析发现共识机制设计模式",
    current_action="使用 search_memory 工具检索",
    current_results="找到 3 篇相关提案",
    title="共识机制设计模式分析",
    memory_type="evidence",
)

# 第2步：提交给 Orchestrator 进行多智能体验证
orchestrator.verify_and_commit(proposal)
```

## 本地创建的核心方法

### ProposalBuilder.create_proposal

核心工厂方法，将非结构化输入规范化为 MemoryProposal：

- 自动生成 proposal_id (UUID4)
- 自动计算 body_hash (SHA-256)
- 自动填充时间戳

### ProposalBuilder.self_verify_proposal

在本地对提案四个维度进行评分，生成 SelfVerification 记录。

### ProposalBuilder.format_proposal_for_agent

将提案格式化为可读字符串，方便 Agent 自行审查并调整自验证评分。
