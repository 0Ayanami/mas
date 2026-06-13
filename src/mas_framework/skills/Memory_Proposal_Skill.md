# Memory Proposal Workflow Skill
## 概览
Agent 在完成一轮完整的 ReAct 循环（Observation -> Thought -> Action -> Observation）后，若有值得共享的记忆沉淀，应当通过本地创建 **Memory Proposal** 并提交至 Orchestrator 进行多智能体验证与共识决策。

## 工作流

### 1. 何时提出 Proposal
在每轮 ReAct 循环结束时，判断以下任意条件：

- **research_note**：得出新的推论、假设或总结性见解
- **evidence**：收集到对当前任务有支持作用的事实数据
- **milestone**：完成了一个可独立提交的子目标
- **tool_observation**：使用外部工具后获得有价值的观察

其中任一条件成立,Agent显示的返回一个Json格式的结果:
```
{
    "signal":"MEMORY_PROPOSE"
    "memory_type":Literal["research_note", "evidence", "milestone", "tool_observation"]
}
```

### 2. 如何构建 Proposal
#### Proposal Header Construction
由Orchestrator按照规则进行构建.

#### Proposal Body Construction
Agent按照Json格式构建Proposal Body:
```
{
    "Thoughts"：thoughts abstract(思考路径关键信息摘要), key decision points & decision results(涉及到的主要决策点信息和决策结果摘要)
    "Action"：action list(执行的操作列表，如action_1: api function call...; action_2: web sesearch with keywords...; action_3: interaction with agent...)
    "Data"：data list(任务相关的关键信息/数据列表，agent本地检索到的提供关键信息摘要，公开渠道获取的提供关键信息摘要和访问链接等)
    "Observations"：result list(当前取得的主要结果或观测情况列表，如result_1: complete subtask_i...; result_2: fetched data from url...)
    "Proposal_Summary": 对整个记忆提案进行摘要
}
```
prompt参考文档：```src\mas_framework\prompts\body_construction_prompt.md```

#### Proposal Self-Verification
Agent针对Proposal在四个维度上（veracity、rationality、value、security）进行自我评分，返回一个Json格式的结果:
```
{
  "veracity": true or false,
  "rationality": true or false,
  "value": true or false,
  "security": true or false,
  "rationale": "string explaining the judgement"
}
```
prompt参考文档：```src\mas_framework\prompts\verify_prompt.md```

### 3. 提交验证（共识阶段）
通过self-verification的proposal会传给 Orchestrator 的 `verify_and_commit` 方法,进行多智能体验证。

- 其他 agent 作为验证者对 proposal 进行多维验证，生成 *VerificationVector*
- *SmartQuorumPolicy* 收集验证结果，计算加权赞成率
- 若赞成率 >= 阈值，提案被 *ACCEPTED* 并持久化到共享记忆
- 否则被 *REJECTED*

### 4. 完整示例


## 本地创建的核心方法
