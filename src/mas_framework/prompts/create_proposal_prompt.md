Build the Body for a MemoryProposal after a completed ReAct cycle.
Return JSON ONLY with this schema:
{% raw %}
{
  "proposal_summary": "one sentence summary",
  "thoughts": {
    "thoughts_abstract": "high-level reasoning summary, no hidden chain-of-thought",
    "key_decisions": [
      {"decision": "what was decided", "result": "adopted|rejected|deferred"}
    ]
  },
  "actions": [
    {"action_id": "act_1", "type": "api_call|tool_call|web_search|agent_interaction|other", "tool": "...", "params": {}, "status": "success|failed|partial"}
  ],
  "data": [
    {"source": "local|web|agent|tool|other", "content_snippet": "key data summary", "url": "", "timestamp": ""}
  ],
  "observations": [
    {"type": "data_retrieval|task_progress|tool_result|other", "description": "...", "status": "complete|partial|failed"}
  ]
}
{% endraw %}

Task context:
{{task_context}}

Completed ReAct trace:
{{react_trace}}
