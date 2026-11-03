"""smolagents and Qwen-Agent for code tasks without a grading tool
(`smolagents_ungraded`, `qwen_agent_ungraded`).

The coding adapters' harmonized prompts name `grade_code` and speak of
grading results. The variants run those adapters unchanged, with only those
prompt pieces replaced for the duration of a run (`patching.swap_globals`).
"""
