"""smolagents and Qwen-Agent for question answering (`smolagents_qa`,
`qwen_agent_qa`), used by GAIA.

The coding adapters' harmonized prompts name `grade_code` and speak of
grading results and programs. The QA variants run those adapters unchanged,
with only their prompt pieces replaced for the duration of a run
(`frameworks/ungraded/patching.swap_globals`), and afterwards hand the
benchmark every tool call of the run, for its grounding labels.
"""
