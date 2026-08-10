# Citations audit

Status: **NOT YET VERIFIED AGAINST SOURCES.** This file exists so the audit is a
checklist rather than an intention. Every entry below was extracted mechanically from
`PAPER_DRAFT.md`, together with the sentence that attributes a claim to it. Nothing here
has been checked against the actual paper.

**This must be completed before submission, not after the results settle.** The risk is
not mainly a wrong year; it is the fourth box — attributing a claim the cited work does
not make. That is the one a reviewer in the area will catch immediately, and it is the one
mechanical extraction cannot help with.

## Known issues found during extraction

1. **`Yao et al.` appears with two years, and both are correct.** 2023 is ReAct
   (reasoning-action-observation loop); 2024 is tau-bench (final-state grading in a
   simulated environment). Two different works by the same first author. Flagged here
   because an automated consistency check reports it as a conflict and it is not one.
2. **`ICLR 2025` is a venue mention, not a citation**, and is matched by any
   author-year regex. Excluded from the list below.
3. Three works are cited in `LITERATURE_REVIEW.md` but not in the draft
   (`Lin et al. 2024`, `Qin et al. 2024`, `Ye et al. 2024`). Two of those — ToolACE/Hammer
   and ToolLLM — are discussed in the draft's §2.2 and §2.6 prose, so the citation may
   have been dropped when text was rewritten. **Check whether the draft makes claims that
   need those citations restored.**
4. Preprint-versus-published drift is the most likely year error in this list, since most
   of these appeared on arXiv before a venue. The cited year must match the version whose
   content is being attributed.
5. **`Healy et al. 2026` needs particular care.** A current-year citation is almost
   certainly a preprint that may have changed since it was read, or may since have been
   published under a different year and venue. Re-read the version being cited.
6. **The extraction is regex-based over parenthetical `(Author, Year)` forms and will have
   missed some.** Multi-author inline forms such as `(Patil, Yan et al., 2025)` for BFCL do
   not match cleanly, and citations written into prose without parentheses are invisible to
   it. A manual pass over the full reference list is still required; this checklist is a
   floor, not a complete inventory.

## Per-citation checklist

### Guo et al. 2024

**Claim as made in the draft:** ...task ultimately succeeds rather than whether each individual call was correct. ToolLLM (Qin et al., 2024) introduced ToolBench, built on thousands of real APIs, but the instability of those live endpoints made results hard to reproduce, which StableToolBench (Guo et al., 2024) addressed by simulating the APIs and caching responses. API-Bank (Li et al., 2023) scored calls by execution match acro...

- [ ] work exists and author list matches
- [ ] year matches the version cited (preprint vs published differ)
- [ ] venue correct
- [ ] **the specific claim above is actually made by this work**

### Healy et al. 2026

**Claim as made in the draft:** ...ves formatting errors by construction, but it does not correct a well-formed call that carries the wrong value. Self-reflection and retry loops attempt to catch errors after execution, and a more recent line of work, including the paper anchoring this project (Healy et al., 2026), detects bad tool selection from a model's internal representations at inference time rather than trying to prevent it...

- [ ] work exists and author list matches
- [ ] year matches the version cited (preprint vs published differ)
- [ ] venue correct
- [ ] **the specific claim above is actually made by this work**

### Li et al. 2023

**Claim as made in the draft:** ...LLM (Qin et al., 2024) introduced ToolBench, built on thousands of real APIs, but the instability of those live endpoints made results hard to reproduce, which StableToolBench (Guo et al., 2024) addressed by simulating the APIs and caching responses. API-Bank (Li et al., 2023) scored calls by execution match across a smaller, controlled set of tools. The Berkeley Function Calling Leaderboard (P...

- [ ] work exists and author list matches
- [ ] year matches the version cited (preprint vs published differ)
- [ ] venue correct
- [ ] **the specific claim above is actually made by this work**

### Liu et al. 2025

**Claim as made in the draft:** ...t call has entered it. That is the gap the propagation framing addresses. Proposed fixes sit at different points in the pipeline. Reliability alignment (Relign) trains models to abstain or clarify when they are unsure. Data-centric approaches such as ToolACE (Liu et al., 2025) and Hammer (Lin et al., 2024) improve function-calling accuracy through synthesized training data and by masking functi...

- [ ] work exists and author list matches
- [ ] year matches the version cited (preprint vs published differ)
- [ ] venue correct
- [ ] **the specific claim above is actually made by this work**

### Maekawa et al. 2025

**Claim as made in the draft:** ...ll sharply once the environment is perturbed. MTU-Bench (Wang et al., 2025) reports tool-selection and parameter accuracy directly and computes them without relying on a language model as judge, which makes the scores cheaper and more consistent. FuncBenchGen (Maekawa et al., 2025) generates contamination-free multi-step tasks as dependency graphs and finds that capable models often produce syntacti...

- [ ] work exists and author list matches
- [ ] year matches the version cited (preprint vs published differ)
- [ ] venue correct
- [ ] **the specific claim above is actually made by this work**

### Patil et al. 2023

**Claim as made in the draft:** ...int, and with what arguments. ReAct (Yao et al., 2023) established the reasoning-action-observation loop that most agent frameworks still follow, and Reflexion (Shinn et al., 2023) added a self-critique step that later reliability work would build on. Gorilla (Patil et al., 2023) was among the first to measure API-call correctness at scale and to treat hallucinated calls as a distinct error to be...

- [ ] work exists and author list matches
- [ ] year matches the version cited (preprint vs published differ)
- [ ] venue correct
- [ ] **the specific claim above is actually made by this work**

### Schick et al. 2023

**Claim as made in the draft:** ...its arguments incorrectly. This section surveys how the field has studied that failure, what it has measured, and where the measurement remains incomplete. ### 2.1 Foundations The idea of teaching a model when and how to call an API goes back to Toolformer (Schick et al., 2023), which learned in a self-supervised way which tool to use, at what point, and with what arguments. ReAct (Yao et al., 2...

- [ ] work exists and author list matches
- [ ] year matches the version cited (preprint vs published differ)
- [ ] venue correct
- [ ] **the specific claim above is actually made by this work**

### Shinn et al. 2023

**Claim as made in the draft:** ...olformer (Schick et al., 2023), which learned in a self-supervised way which tool to use, at what point, and with what arguments. ReAct (Yao et al., 2023) established the reasoning-action-observation loop that most agent frameworks still follow, and Reflexion (Shinn et al., 2023) added a self-critique step that later reliability work would build on. Gorilla (Patil et al., 2023) was among the first...

- [ ] work exists and author list matches
- [ ] year matches the version cited (preprint vs published differ)
- [ ] venue correct
- [ ] **the specific claim above is actually made by this work**

### Wang et al. 2025

**Claim as made in the draft:** ...evaluates tool selection, parameter identification, and content filling as separate stages, and does so under increasing levels of noise, showing that accuracy that looks solid in clean conditions can fall sharply once the environment is perturbed. MTU-Bench (Wang et al., 2025) reports tool-selection and parameter accuracy directly and computes them without relying on a language model as judge,...

- [ ] work exists and author list matches
- [ ] year matches the version cited (preprint vs published differ)
- [ ] venue correct
- [ ] **the specific claim above is actually made by this work**

### Xu et al. 2025

**Claim as made in the draft:** ...eded, or skipping a call that was required. - **Argument errors:** malformed or unparseable output, missing required parameters, wrong types, wrong values, and fabricated arguments supplied instead of asking the user for the missing information. Relign (Xu et al., 2025) formalizes this split into tool-selection and tool-usage hallucination and adds the option for a model to defer or ask...

- [ ] work exists and author list matches
- [ ] year matches the version cited (preprint vs published differ)
- [ ] venue correct
- [ ] **the specific claim above is actually made by this work**

### Yao et al. 2023

**Claim as made in the draft:** ...asurement remains incomplete. ### 2.1 Foundations The idea of teaching a model when and how to call an API goes back to Toolformer (Schick et al., 2023), which learned in a self-supervised way which tool to use, at what point, and with what arguments. ReAct (Yao et al., 2023) established the reasoning-action-observation loop that most agent frameworks still follow, and Reflexion (Shinn et al.,...

- [ ] work exists and author list matches
- [ ] year matches the version cited (preprint vs published differ)
- [ ] venue correct
- [ ] **the specific claim above is actually made by this work**

### Yao et al. 2024

**Claim as made in the draft:** ...Calling Leaderboard (Patil, Yan et al., 2025) has become the de facto standard, combining abstract-syntax-tree matching with executable checks and expanding over successive versions from single calls to parallel, multi-turn, and multi-step settings. tau-bench (Yao et al., 2024) took a different route, grading the final state of a simulated environment after an agent-user dialogue and reporting r...

- [ ] work exists and author list matches
- [ ] year matches the version cited (preprint vs published differ)
- [ ] venue correct
- [ ] **the specific claim above is actually made by this work**
