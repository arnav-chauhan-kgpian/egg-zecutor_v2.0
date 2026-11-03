# hallu_harness — one model, several agent frameworks, several datasets

Runs a model as an agent under smolagents, Qwen-Agent or a plain-text
protocol, on Codeforces or HumanEval, and records every step in one
trajectory format built for hallucination research. The full design, and
the reasons behind each choice, are in `DESIGN.md` at the repo root; this
file is the working reference.

    pip install -r requirements-harness.txt
    python -m hallu_harness run --config configs/frameworks/smolagents.yaml \
        --set model=qwen2.5-coder-7b --set 'dataset.selection.ids=[2019/A]'

## What is held fixed across frameworks

Comparing frameworks only means something if everything else is equal, so
these are shared:

* **One model engine.** Every framework calls the harness's own backend
  (`models/`), never its own model class. The sampler, the decoding settings
  and the logits-processor chain are the same everywhere, and the engine
  checks at load time that transformers adds nothing of its own.
* **One task text.** With `prompt_mode: harmonized` (the default) every
  framework gets the same task instructions and user message. Only the
  phrases that say *how to reply* differ: call a tool, or write a
  ```` ```python ```` block (`core/protocol.py`). `prompt_mode: stock` uses
  each framework's own system prompt instead. The task instructions are
  rendered from the run's own settings (reply budget, hidden tests on or
  off, review turn, solution policy, time limits), so every statement in
  them is true of the run; a config they could not describe truthfully is
  refused (hidden tests with a policy that does not start with `accepted`,
  or the `submit` tool without `submitted` in the policy).
* **One tool.** `grade_code` runs the program against the visible tests and,
  only if all pass, against the hidden tests. It tells the model counts and
  failure types, never a hidden input or output. Every sentence it says
  comes from the dataset's `Texts`.
* **No truncation.** Nothing ever shortens the conversation. If the next
  prompt plus `max_new_tokens` would not fit the model's window, the
  trajectory ends with `stop_reason: context_exhausted`.

## Configuration

A run is one framework config plus presets it points to:

    configs/
      frameworks/{smolagents,qwen_agent,text}.yaml   the framework, its params, the run settings
      models/*.yaml                                   weights, backend, decoding (shared by all frameworks)
      datasets/{codeforces,humaneval}.yaml            task selection, grading, which tools the dataset needs

Any key can be overridden with `--set`. Values are YAML, and a preset name
swaps the whole preset:

    --set model=qwen2.5-coder-32b
    --set dataset=humaneval
    --set dataset.selection.ids=[HumanEval/0,HumanEval/12]
    --set model.generation.temperature=0.7
    --set params.max_model_calls=8
    --set run.samples_per_task=5

Every section rejects unknown keys, so a typo is an error, not a silently
ignored setting. Before a long run, check what will actually run:

    python -m hallu_harness check --config configs/frameworks/qwen_agent.yaml --set dataset=humaneval --count-tasks
    python -m hallu_harness diff configs/frameworks/smolagents.yaml configs/frameworks/qwen_agent.yaml

API keys are never written into a config. A model preset names the
environment variable that holds the key (`api.api_key_env`).

## Output

    trajectories/<run_id>/
      run.json                     resolved config, versions, dataset and exclusion-file hashes
      index.jsonl                  one line per finished trajectory: solved, stop reason, counts
      tasks/<task>__s<k>.json      the trajectory (schema v2)
      tasks/<task>__s<k>.tokens.npz  token ids, log-probs, entropy, top-k per step
    logs/<framework>/<dataset>/<model>/<YYYY-MM-DD_HH-MM-SS>.log
                                   everything the run printed, stdout and stderr

`<run_id>` is `<UTC start time>-<host>-<framework>-<dataset>-<model>`, e.g.
`20261014T081031Z-gnode3-smolagents-codeforces-Qwen2.5-Coder-7B-Instruct`.
The log is named by the date and time the run started on the machine's clock,
e.g. `logs/smolagents/codeforces/Qwen2.5-Coder-7B-Instruct/2026-10-14_13-40-31.log`
(`_2`, `_3`, ... if another run took that second), and its `[log]` line names
the run id. It is a copy of the terminal for the whole command:
config, model loading, warnings, every step, and the traceback if the run
dies. `run.log_dir` sets the top directory (`null` turns the log off);
`run.log_file: PATH` appends every run to one file instead.

Each trajectory holds the system prompt, the user query, dataset, model and
framework metadata, and per step:

* the exact model input (the templated string the weights saw);
* the raw output, special tokens included;
* whether the reply hit the length cap;
* the parse status;
* every tool call with its arguments, the observation the model saw, and the
  private grading detail it did not see (per-test results, hidden verdict,
  failing test numbers);
* segment spans (system, task, observation, think, tool call, code, …) over
  the input and output tokens.

A trajectory is written atomically when it ends, with a `.partial.json`
checkpoint after every step until then. Runs resume: a trajectory already
complete under the same experiment key is skipped. The key covers the
config, the grading data, `PROMPT_REVISION` in `core/protocol.py`, and the
model and framework as actually loaded: the resolved weights and tokenizer
revisions, the chat template, dtype, the decoding settings in effect, and
the framework and transformers versions. Bump `PROMPT_REVISION` whenever
text the model is shown changes, so a resumed run never mixes wordings.

A failure of the harness itself — fetching the task's test data, inside a
tool, grading the final solution — is never shown to the model: the
trajectory ends with `status: crashed` and is re-run on resume. A failed
model call, including one a framework swallows, gives `status: error`
(listed in `outcome.model_errors`) and is re-run too.

`run.verbosity: verbose` prints the exact model input and output for every
call. `run.verbose_input: delta` prints only what is new since the previous
call, and `full` prints all of it.

## Datasets

**Codeforces** (`open-r1/codeforces`, test split). Only problems with fixed
answers and generated hidden tests are eligible. 13 multi-answer problems
that ship no checker are excluded (`benchmarks/codeforces/excluded_tasks.json`).
Outputs are compared as open-r1 does: whitespace-separated tokens, with
YES/NO case-insensitive. The model is shown its output up to the expected
output's token count, and a program that starts printing more is stopped
and counts as a wrong answer.

**Time limits** (both datasets) are on CPU time, as a judge measures them,
so a busy machine cannot turn a fast-enough program into a timeout. A test
that waits instead of computing is stopped after 2 × its limit + 1 s of real
time. A test that went over either limit is TIME_LIMIT_EXCEEDED however it
ended. Programs run with one BLAS/OpenMP thread, so numerical libraries load
within the memory cap.

**HumanEval** (`openai/openai_humaneval` tasks, `evalplus/humanevalplus`
tests). The docstring examples are the visible tests; HumanEval+'s inputs
are the hidden tests, each under its own time limit of 4× the reference
solution's CPU time, with a floor of 1 s; all of them run in one process,
stopped after 600 s of real time (the instructions say so). A failed
example shows the expected
and the returned value exactly as `repr()` prints them, whitespace
untouched; the returned one is cut 50 characters past the expected one's
length. A task is kept only when its
docstring examples, the original tests and the HumanEval+ tests all agree
with the reference solution, and the original HumanEval reference passes the
HumanEval+ tests within the grading limits. `audit-humaneval` checks this
and writes
`benchmarks/humaneval/excluded_tasks.json` and the reference timings:

    python -m hallu_harness audit-humaneval --workers 8

A task requested by id that is excluded is skipped with a `[skip]` line on
stderr.

**GAIA** (`gaia-benchmark/GAIA`, gated, validation split at a pinned commit;
the test split's answers are not public). It is a question-answering
benchmark with no grading tool. It runs under `smolagents_qa` or
`qwen_agent_qa`, which are the coding adapters with question-answering
prompts; the coding prompts name `grade_code`.

    hf auth login                                    # an account that accepted GAIA's terms
    export SERPER_API_KEY=...                        # web_search's default backend (never put the key in a config)
    python -m hallu_harness gaia-data                # download; count the questions the tools serve
    python -m hallu_harness run --config configs/frameworks/smolagents_gaia.yaml --set 'dataset.selection.levels=[1]'
    python -m hallu_harness run --config configs/frameworks/qwen_agent_gaia.yaml

The tools are text only:

| tool | what it does |
|---|---|
| `web_search` (registry name `gaia_web_search`) | Google's results through Serper by default (key in `$SERPER_API_KEY`); once Serper's credits are spent, the free engines, each result saying so (`fallback: none` crashes instead); `backend: ddgs` uses only the free engines; titles and snippets are cut |
| `visit_webpage` | HTML, PDF and text, shown one page at a time |
| `find_in_page` | finds a phrase in a page or the attachment and gives the page numbers `visit_webpage` and `read_file` show |
| `read_file` | the attachment: xlsx as tables with Excel coordinates, PDF, text files |
| `python` | the sandbox jail, with the attachment in the working directory |

**Questions served.** By default only the questions those tools can serve
are run: 114 of 165. The rules are in `benchmarks/gaia/capabilities.py`,
with hand-checked exceptions in `capability_overrides.json`.

**Page size.** A page is at most `page_chars` units: digits, symbols and
non-Latin characters cost more than letters. Measured with Qwen3's tokenizer
on 25 sites and every attachment, no page passed 2,650 tokens.

**Grading.** The answer, read as `core/qa_protocol.py` tells the model, is
scored with GAIA's own scorer. The grade adds grounding labels: which steps'
observations contain the reference or the model's answer, and which URLs the
model visited without having been shown them.

The reference answer is never written to a trajectory, only its SHA-256.
GAIA's terms forbid resharing the answers.

## Frameworks

| framework | how the model acts | ends when |
|---|---|---|
| `smolagents` | `ToolCallingAgent`, JSON action blobs, stock parser | `final_answer`, or the step budget |
| `qwen_agent` | `FnCallAgent`, `<tool_call>` tags (`nous`) | a reply with no tool call, or `MAX_LLM_CALL_PER_RUN` |
| `text` | one ```` ```python ```` block per reply, routed to `grade_code` | the reply `DONE` / `SORRY`, or the budget |

Every framework makes at most `max_model_calls` model calls.
smolagents' stock extra "answer from memory" call after the budget is off
by default (`answer_from_memory_on_budget`): it sees the last grading
result, so it would be one more informed attempt than the others get.

smolagents records each past call in the history as `Calling tools:` followed
by the call. Stock smolagents appends this to the model's own reply, and the
model then learns to write it itself. Here it opens the next user turn
instead, before `Observation:` (`calling_tools_turn: user`);
`calling_tools_turn: assistant` restores the stock rendering.

Nothing the model is shown should mislead it about the setup, so a few other
stock behaviours are changed, each with a switch or a note in `DESIGN.md`
§9.1–9.4: smolagents' tool schemas no longer also go to the chat template
(`chat_template_tools`), whose own tools section contradicted smolagents'
reply rules; its prompt shows the form of a reply instead of a worked example
the model copied; its JSON errors quote the part of the JSON that failed;
Qwen-Agent's tool errors leave out the traceback through the harness's code;
and the text protocol finishes with the reply `DONE`, not a code block
holding `print("DONE")`.

Stop reasons mean the same in every framework: `final_answer` when the
model finished or gave up as its protocol says; `budget`;
`context_exhausted`; `error`. Two are framework-specific: smolagents'
`no_final_answer`, and Qwen-Agent's `no_tool_call`, when its run ended on a
reply whose tool call did not parse or that hit the length cap. A tool call
with broken JSON is a parse status `failed` in every framework, even where
Qwen-Agent's lenient extractor still makes a call of it.

Whatever the framework, the trajectory's answer is the first program that
passed the hidden tests. If none did, it is the code in the final answer,
and failing that the last program submitted (`solution_policy`).

## Interpretability

Hidden states are not saved during generation. They are recomputed offline
from the recorded token ids, one teacher-forced pass per step:

    python -m hallu_harness replay --run trajectories/<run_id> --out activations/<name> \
        --layers quarters --segments observation,think,tool_call [--lookback]

Before writing anything, the replay checks itself against the log-probs
recorded at generation time. A step is rejected if more than 5% of its tokens
(and more than two) differ by more than 0.1 nats, or if any token differs by
more than 2 nats. bf16 rounding moves a few tokens, some a long way; a wrong
model moves many (`interp/extract.py`, `logprob_check`). It also
refuses to run under a different model revision, dtype, torch or
transformers unless forced. Only `hf_local` and `vllm` runs record token ids;
API runs cannot be replayed.

Logit-lens readouts (DESIGN.md §11.5) use the same replay and checks. With
`run.lens.enabled` (on in the framework configs) the run computes them itself,
for each step as soon as its model call returns, into
`trajectories/<run_id>/lens/`. For a run made without it, or to finish steps a
run could not compute (out of GPU memory, disk below `run.lens.min_free_gb`):

    python -m hallu_harness lens --run trajectories/<run_id>      # writes trajectories/<run_id>/lens/

For every context and generated token, and every layer, this stores four
readouts (attention output, mid-layer residual, MLP output, layer output):
each readout's top-5 token ids, their probabilities, and its entropy. Read a
step back with `hallu_harness.interp.lens.load_lens(path).step(k)`.

## Other commands

    python -m hallu_harness export --run trajectories/<run_id> --format v1 --out analysis/runs_v1
    python -m hallu_harness export --run trajectories/<run_id> --format agenthallu --out exports/
    python -m hallu_harness list frameworks|benchmarks|tools|backends

`v1` is cf_agent's old record format, so `analysis/traj_lib.py` and the
existing notebooks run unchanged. `traj_lib.load_run_v2` also reads v2 runs
directly.

## Adding a framework, dataset, tool or backend

Each is a class registered by name in `core/registry.py`, by import path, so
a config imports only what it names:

* **Framework.** A `FrameworkAdapter` (`frameworks/base.py`) with a pydantic
  `Params` model and a `Protocol`. It must route every model call through the
  shared backend and report each event to the `StepRecorder`.
* **Dataset.** A `Benchmark` (`benchmarks/base.py`) with an `Evaluator`
  (visible and hidden tests) and a `Texts` (every sentence the model is
  shown), plus a preset in `configs/datasets/`.
* **Tool.** A `Tool` (`tools/base.py`): name, JSON-Schema parameters,
  `call()`. Each framework bridges it.

The conformance suite (`tests/test_conformance*.py`) runs scripted replies
through every framework and states what every trajectory must record. A new
framework should pass it unchanged.

## Tests

    python -m pytest                      # everything
    python -m pytest -m "not model"       # skip the tests that load a small real model

Tests marked `network` need the Hugging Face datasets in the local cache.
