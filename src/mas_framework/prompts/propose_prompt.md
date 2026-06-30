You have completed one ReAct cycle in a multi-agent task.

Decide whether the result should be proposed as shared memory.
Return JSON ONLY:
{% raw %}
{
  "propose_memory": true or false,
  "signal": "MEMORY_PROPOSE" or "NO_MEMORY",
  "memory_type": "research_note" | "evidence" | "milestone" | "tool_observation",
  "rationale": "short reason"
}
{% endraw %}

Task context:
{{task_context}}

Completed ReAct trace:
{{react_trace}}

Memory proposal guidance:
{{memory_proposal_skill}}
