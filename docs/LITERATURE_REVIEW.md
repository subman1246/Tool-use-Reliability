## Literature Review

Large language models are increasingly deployed as agents that call external tools and APIs to act in the world, and the point at which they do so has become a common source of failure. A model can reason correctly about what needs to happen and still produce a call that fails, either by choosing the wrong tool or by filling its arguments incorrectly. This review looks at how the field has studied that failure, what it has measured, and where the measurement remains incomplete. Our focus is the reliability of the invocation itself, meaning whether a model selects the correct tool and supplies valid, correct arguments, and how that reliability varies from one model to another.

### Foundations

The idea of teaching a model when and how to call an API goes back to Toolformer (Schick et al., 2023), which learned in a self-supervised way which tool to use, at what point, and with what arguments. ReAct (Yao et al., 2023) established the reasoning-action-observation loop that most agent frameworks still follow, and Reflexion (Shinn et al., 2023) added a self-critique step that later reliability work would build on. Gorilla (Patil et al., 2023) was among the first to measure API-call correctness at scale and to treat hallucinated calls as a distinct error to be counted rather than an occasional nuisance. Together these define the pipeline the rest of the literature examines: pick a tool, then form its arguments.

### Benchmarks and what they measure

Most large tool-use benchmarks evaluate whether a task ultimately succeeds rather than whether each individual call was correct. ToolLLM (Qin et al., 2024) introduced ToolBench, built on thousands of real APIs, but the instability of those live endpoints made results hard to reproduce, which StableToolBench (Guo et al., 2024) addressed by simulating the APIs and caching responses. API-Bank (Li et al., 2023) scored calls by execution match across a smaller, controlled set of tools. The Berkeley Function Calling Leaderboard (Patil, Yan et al., 2025) has become the de facto standard, combining abstract-syntax-tree matching with executable checks and expanding over successive versions from single calls to parallel, multi-turn, and multi-step settings. tau-bench (Yao et al., 2024) took a different route, grading the final state of a simulated environment after an agent-user dialogue and reporting reliability across repeated trials, where even strong models succeed on fewer than half of tasks and behave inconsistently from run to run.

The common thread is that these benchmarks report an aggregate number. That number is useful for ranking models, but it does not say where a call went wrong, and it usually reflects clean conditions in which the tools behave as expected.

### Localizing the failure

A smaller group of benchmarks breaks the call apart. RoTBench (Ye et al., 2024) evaluates tool selection, parameter identification, and content filling as separate stages, and does so under increasing levels of noise, showing that accuracy that looks solid in clean conditions can fall sharply once the environment is perturbed. MTU-Bench (Wang et al., 2025) reports tool-selection and parameter accuracy directly and computes them without relying on a language model as judge, which makes the scores cheaper and more consistent. FuncBenchGen (Maekawa et al., 2025) generates contamination-free multi-step tasks as dependency graphs and finds that capable models often produce syntactically valid calls while carrying incorrect or stale argument values from one step to the next, with performance degrading as the chain of dependencies grows. This work is the closest in spirit to an invocation-level view, though it still reports task success rather than a single disaggregated measure of call correctness.

### How the field categorizes errors

Across these studies the failures sort into two families:

- **Selection errors:** choosing the wrong tool, inventing a tool that does not exist, calling a tool when none was needed, or skipping a call that was required.
- **Argument errors:** malformed or unparseable output, missing required parameters, wrong types, wrong values, and fabricated arguments supplied instead of asking the user for the missing information.

Relign (Xu et al., 2025) formalizes this split into tool-selection and tool-usage hallucination and adds the option for a model to defer or ask for clarification rather than guess.

### Error propagation in multi-step chains

Selection and argument errors are stateless: they describe a single call in isolation. In a multi-step task, where the arguments of one call are drawn from the outputs of earlier calls, a further structure appears that the two families do not capture. A single mistake at step t does not simply lower the score for that step; it changes the inputs every downstream step receives, so later calls can be locally well-formed and still globally wrong. This is the behaviour FuncBenchGen (Maekawa et al., 2025) isolates when it reports models carrying stale or incorrect values forward, and it is worth separating out rather than folding into the argument family.

The distinction that makes this tractable is between two notions of a correct invocation. A call is locally correct if, given the correct inputs, the model selects the right tool and forms valid, correct arguments. A call is globally correct if it is correct inside the model's own run, where its inputs may already be poisoned by an upstream error. A model can be strong locally and still lose accuracy across a chain if it cannot recover once the context is corrupted. The gap between the two is the quantity of interest for multi-step reliability, and the propagation itself takes a small number of recognisable forms: a wrong value at step t consumed as an argument at t+1; a malformed output that breaks parsing for everything downstream; a wrong tool choice that sends the plan down a branch where subsequent calls are off-target even when individually valid; and repeated or looping calls when a result is never registered.

Framed this way, the descriptive finding that accuracy falls with dependency depth becomes something that can be modelled. If per-step errors were independent, the probability that a chain of length n is entirely correct would decay as p^n for a per-step reliability p. The propagation effect is precisely the departure from that baseline: once a step is wrong, the reliability of the next step drops from its clean value p to a lower poisoned value, and the size of that drop is a property of the model that can be estimated. The methodology develops this into a fitted decay model with a single interpretable poisoning parameter, which is what lets the study compare brittleness across models rather than only reporting that everyone gets worse with depth.

### Methods for improving reliability

Proposed fixes sit at different points in the pipeline. Reliability alignment (Relign) trains models to abstain or clarify when they are unsure. Data-centric approaches such as ToolACE (Liu et al., 2025) and Hammer (Lin et al., 2024) improve function-calling accuracy through synthesized training data and by masking function and parameter names so models generalize beyond surface patterns rather than memorizing them. Constrained or schema-guided decoding removes formatting errors by construction, but it does not correct a well-formed call that carries the wrong value. Self-reflection and retry loops attempt to catch errors after execution, and a more recent line of work, including the paper anchoring this project (Healy et al., 2026), detects bad tool selection from a model's internal representations at inference time rather than trying to prevent it during training.

### Metrics and the gap this study addresses

Evaluation across this literature uses exact match on the full call, tool-selection accuracy, parameter-level accuracy, abstract-syntax-tree matching, and success across repeated trials. A recurring observation is that tool-selection accuracy alone is necessary but not sufficient: a model that picks the right tool but passes a wrong argument scores well on selection and still fails the user. This is the reasoning behind measuring the two components together.

Three gaps follow from the review:

- Headline benchmarks aggregate their results and rarely localize where invocation fails, and the studies that do disaggregate tend to cover few models or non-fixed task sets.
- Cross-model comparisons that separate selection errors from argument errors are uncommon, and they are reported on different tasks with different metrics, so no clean and comparable picture exists.
- Most evaluation reflects a happy path, while the work that stresses models with noise, distractors, and dependency depth shows reliability dropping in ways a clean score hides.

This study addresses those gaps by measuring a correct-invocation rate, defined as selecting the correct tool and supplying valid, correct arguments, on a fixed set of multi-step tool-use tasks, and by reporting that rate at both the local and global level, broken down by error type and by dependency depth. Rather than testing an open-ended set of whatever models are available, it uses a suite designed to vary one factor at a time: a single open-weight family evaluated across parameter scales to isolate the effect of scale, several families compared at a matched scale to isolate architecture and training, function-calling-tuned variants set against their general instruct bases to isolate the effect of that tuning, and one or two proprietary frontier models as a reference ceiling. On top of this, the propagation of errors across depth is fitted to a decay model with a single poisoning parameter, so the study can say not only which models are more reliable but how brittle each is once its own context has been corrupted. The full suite and the model are specified in the methodology. The aim is not another end-to-end ranking but a comparable, invocation-level account of where and how tool calls fail.

### References

Chen et al. (2025). ACEBench: A comprehensive evaluation of tool usage. Findings of EMNLP.

Guo et al. (2024). StableToolBench: Towards stable large-scale benchmarking of tool learning. Findings of ACL.

Healy et al. (2026). Internal representations as indicators of hallucinations in agent tool selection.

Li et al. (2023). API-Bank: A comprehensive benchmark for tool-augmented LLMs. EMNLP.

Lin et al. (2024). Hammer: Robust function-calling for on-device language models via function masking.

Liu et al. (2025). ToolACE: Winning the points of LLM function calling. ICLR.

Maekawa et al. (2025). Towards reliable benchmarking: A contamination-free, controllable evaluation framework for multi-step LLM function calling (FuncBenchGen).

Patil et al. (2023). Gorilla: Large language model connected with massive APIs.

Patil, Yan et al. (2025). The Berkeley Function Calling Leaderboard (BFCL): From tool use to agentic evaluation. ICML.

Qin et al. (2024). ToolLLM: Facilitating large language models to master 16000+ real-world APIs. ICLR.

Schick et al. (2023). Toolformer: Language models can teach themselves to use tools. NeurIPS.

Shinn et al. (2023). Reflexion: Language agents with verbal reinforcement learning. NeurIPS.

Wang et al. (2025). MTU-Bench: A multi-granularity tool-use benchmark for large language models. ICLR.

Xu et al. (2025). Reducing tool hallucination via reliability alignment. ICML.

Yao et al. (2023). ReAct: Synergizing reasoning and acting in language models. ICLR.

Yao et al. (2024). tau-bench: A benchmark for tool-agent-user interaction in real-world domains. ICLR 2025.

Ye et al. (2024). RoTBench: A multi-level benchmark for evaluating the robustness of large language models in tool learning. EMNLP.
