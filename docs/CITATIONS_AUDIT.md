# Citations audit

Status: **NOT YET VERIFIED AGAINST SOURCES.** Every entry below was extracted
mechanically from `PAPER_DRAFT.md` together with the sentence that attributes a claim to
it. Nothing here has been checked by reading the cited work. This file exists so the audit
is a checklist rather than an intention, and it should stay marked unverified until someone
has actually read each paper against the sentence citing it.

**Complete this before submission, not after the results settle.** The risk is not mainly a
wrong year; it is the fourth box on each entry — attributing a claim the cited work does
not make. That is what a reviewer in the area catches immediately, and it is the one thing
mechanical extraction cannot help with.

## A methods point about auditing citations

**Automated extraction is a floor and needs manual cross-validation; one apparent gap here
was a parsing artifact, not a missing citation.** A regex over `(Author, Year)` forms
reported three works as discussed-but-uncited, which would have been a genuine defect. They
were cited; the citations wrapped across line breaks and the pattern required a literal
space. Acting on the automated output would have duplicated three citations while
"fixing" nothing.

The general lesson for anyone auditing this way: an extraction tool's false negatives
become false *positives* about missing citations, which read as findings rather than as
tool failures. Cross-validate against the prose before acting on any gap it reports.

## Inventory

**16 distinct citations, and every one appears in both `PAPER_DRAFT.md` and
`LITERATURE_REVIEW.md`. There are no orphaned works.**

### A correction to an earlier version of this audit

An earlier pass reported that three works (`Lin et al. 2024`, `Qin et al. 2024`,
`Ye et al. 2024`) were discussed in the draft's prose but had lost their citations —
uncited-claims territory. **That was wrong, and it was my extraction that was at fault.**
The citations are present; they are wrapped across line breaks (`Ye et al.,
2024`,
`Lin et
al., 2024`) and the first regex used a literal space, so it did not match them.
The same bug hid `Patil, Yan et al. 2025` for BFCL, which the earlier pass listed as
unmatchable.

Two things follow. Restoring those citations would have **duplicated** them, so the check
was worth doing before acting. And the caveat that the regex is a floor rather than an
inventory was correct in substance but understated: it did not merely miss citations, it
produced a false positive about missing ones. **The manual pass over the reference list is
the actual audit; this checklist is scaffolding for it.**

### Other notes from extraction

1. **`Yao et al.` appears with two years, and both are correct.** 2023 is ReAct
   (reasoning-action-observation loop); 2024 is tau-bench (final-state grading in a
   simulated environment). Two works, same first author. An automated consistency check
   flags this as a conflict; it is not one.
2. **`ICLR 2025` is a venue mention, not a citation**, and is matched by any author-year
   regex. Excluded below.
3. **`Healy et al. 2026` needs particular care.** A current-year citation is almost
   certainly a preprint that may have moved since it was read, or may since have been
   published under a different year and venue.
4. Preprint-versus-published drift is the most likely year error here, since most of these
   appeared on arXiv before a venue. The cited year must match the version whose content is
   being attributed.

## Per-citation checklist

### Guo et al. 2024

**Claim as made in the draft:** ...task ultimately succeeds rather than whether each individual call was correct. ToolLLM (Qin et al., 2024) introduced ToolBench, built on thousands of real APIs, but the instability of those live endpoints made results hard to reproduce, which StableToolBench (Guo et al., 2024) addressed by simulating the APIs and caching responses. API-Bank (Li et al., 2023) scored calls by ex...

- [ ] work exists and author list matches
- [ ] year matches the version cited (preprint vs published differ)
- [ ] venue correct
- [ ] **the specific claim above is actually made by this work**

### Healy et al. 2026

**Claim as made in the draft:** ...ves formatting errors by construction, but it does not correct a well-formed call that carries the wrong value. Self-reflection and retry loops attempt to catch errors after execution, and a more recent line of work, including the paper anchoring this project (Healy et al., 2026), detects bad tool selection from a model's internal representations at inference time rather than t...

- [ ] work exists and author list matches
- [ ] year matches the version cited (preprint vs published differ)
- [ ] venue correct
- [ ] **the specific claim above is actually made by this work**

### Li et al. 2023

**Claim as made in the draft:** ...LLM (Qin et al., 2024) introduced ToolBench, built on thousands of real APIs, but the instability of those live endpoints made results hard to reproduce, which StableToolBench (Guo et al., 2024) addressed by simulating the APIs and caching responses. API-Bank (Li et al., 2023) scored calls by execution match across a smaller, controlled set of tools. The Berkeley Function Calli...

- [ ] work exists and author list matches
- [ ] year matches the version cited (preprint vs published differ)
- [ ] venue correct
- [ ] **the specific claim above is actually made by this work**

### Lin et al. 2024

**Claim as made in the draft:** ...the gap the propagation framing addresses. Proposed fixes sit at different points in the pipeline. Reliability alignment (Relign) trains models to abstain or clarify when they are unsure. Data-centric approaches such as ToolACE (Liu et al., 2025) and Hammer (Lin et al., 2024) improve function-calling accuracy through synthesized training data and by masking function and param...

- [ ] work exists and author list matches
- [ ] year matches the version cited (preprint vs published differ)
- [ ] venue correct
- [ ] **the specific claim above is actually made by this work**

### Liu et al. 2025

**Claim as made in the draft:** ...t call has entered it. That is the gap the propagation framing addresses. Proposed fixes sit at different points in the pipeline. Reliability alignment (Relign) trains models to abstain or clarify when they are unsure. Data-centric approaches such as ToolACE (Liu et al., 2025) and Hammer (Lin et al., 2024) improve function-calling accuracy through synthesized training data and...

- [ ] work exists and author list matches
- [ ] year matches the version cited (preprint vs published differ)
- [ ] venue correct
- [ ] **the specific claim above is actually made by this work**

### Maekawa et al. 2025

**Claim as made in the draft:** ...ll sharply once the environment is perturbed. MTU-Bench (Wang et al., 2025) reports tool-selection and parameter accuracy directly and computes them without relying on a language model as judge, which makes the scores cheaper and more consistent. FuncBenchGen (Maekawa et al., 2025) generates contamination-free multi-step tasks as dependency graphs and finds that capable models...

- [ ] work exists and author list matches
- [ ] year matches the version cited (preprint vs published differ)
- [ ] venue correct
- [ ] **the specific claim above is actually made by this work**

### Patil et al. 2023

**Claim as made in the draft:** ...int, and with what arguments. ReAct (Yao et al., 2023) established the reasoning-action-observation loop that most agent frameworks still follow, and Reflexion (Shinn et al., 2023) added a self-critique step that later reliability work would build on. Gorilla (Patil et al., 2023) was among the first to measure API-call correctness at scale and to treat hallucinated calls as a d...

- [ ] work exists and author list matches
- [ ] year matches the version cited (preprint vs published differ)
- [ ] venue correct
- [ ] **the specific claim above is actually made by this work**

### Patil, Yan et al. 2025

**Claim as made in the draft:** ...hard to reproduce, which StableToolBench (Guo et al., 2024) addressed by simulating the APIs and caching responses. API-Bank (Li et al., 2023) scored calls by execution match across a smaller, controlled set of tools. The Berkeley Function Calling Leaderboard (Patil, Yan et al., 2025) has become the de facto standard, combining abstract-syntax-tree matching with executable chec...

- [ ] work exists and author list matches
- [ ] year matches the version cited (preprint vs published differ)
- [ ] venue correct
- [ ] **the specific claim above is actually made by this work**

### Qin et al. 2024

**Claim as made in the draft:** ...line the rest of the literature examines: pick a tool, then form its arguments. ### 2.2 Benchmarks and what they measure Most large tool-use benchmarks evaluate whether a task ultimately succeeds rather than whether each individual call was correct. ToolLLM (Qin et al., 2024) introduced ToolBench, built on thousands of real APIs, but the instability of those live endpoints ma...

- [ ] work exists and author list matches
- [ ] year matches the version cited (preprint vs published differ)
- [ ] venue correct
- [ ] **the specific claim above is actually made by this work**

### Schick et al. 2023

**Claim as made in the draft:** ...its arguments incorrectly. This section surveys how the field has studied that failure, what it has measured, and where the measurement remains incomplete. ### 2.1 Foundations The idea of teaching a model when and how to call an API goes back to Toolformer (Schick et al., 2023), which learned in a self-supervised way which tool to use, at what point, and with what arguments....

- [ ] work exists and author list matches
- [ ] year matches the version cited (preprint vs published differ)
- [ ] venue correct
- [ ] **the specific claim above is actually made by this work**

### Shinn et al. 2023

**Claim as made in the draft:** ...olformer (Schick et al., 2023), which learned in a self-supervised way which tool to use, at what point, and with what arguments. ReAct (Yao et al., 2023) established the reasoning-action-observation loop that most agent frameworks still follow, and Reflexion (Shinn et al., 2023) added a self-critique step that later reliability work would build on. Gorilla (Patil et al., 2023)...

- [ ] work exists and author list matches
- [ ] year matches the version cited (preprint vs published differ)
- [ ] venue correct
- [ ] **the specific claim above is actually made by this work**

### Wang et al. 2025

**Claim as made in the draft:** ...evaluates tool selection, parameter identification, and content filling as separate stages, and does so under increasing levels of noise, showing that accuracy that looks solid in clean conditions can fall sharply once the environment is perturbed. MTU-Bench (Wang et al., 2025) reports tool-selection and parameter accuracy directly and computes them without relying on a langua...

- [ ] work exists and author list matches
- [ ] year matches the version cited (preprint vs published differ)
- [ ] venue correct
- [ ] **the specific claim above is actually made by this work**

### Xu et al. 2025

**Claim as made in the draft:** ...eded, or skipping a call that was required. - **Argument errors:** malformed or unparseable output, missing required parameters, wrong types, wrong values, and fabricated arguments supplied instead of asking the user for the missing information. Relign (Xu et al., 2025) formalizes this split into tool-selection and tool-usage hallucination and adds the option for a model...

- [ ] work exists and author list matches
- [ ] year matches the version cited (preprint vs published differ)
- [ ] venue correct
- [ ] **the specific claim above is actually made by this work**

### Yao et al. 2023

**Claim as made in the draft:** ...asurement remains incomplete. ### 2.1 Foundations The idea of teaching a model when and how to call an API goes back to Toolformer (Schick et al., 2023), which learned in a self-supervised way which tool to use, at what point, and with what arguments. ReAct (Yao et al., 2023) established the reasoning-action-observation loop that most agent frameworks still follow, and Reflex...

- [ ] work exists and author list matches
- [ ] year matches the version cited (preprint vs published differ)
- [ ] venue correct
- [ ] **the specific claim above is actually made by this work**

### Yao et al. 2024

**Claim as made in the draft:** ...Calling Leaderboard (Patil, Yan et al., 2025) has become the de facto standard, combining abstract-syntax-tree matching with executable checks and expanding over successive versions from single calls to parallel, multi-turn, and multi-step settings. tau-bench (Yao et al., 2024) took a different route, grading the final state of a simulated environment after an agent-user dialog...

- [ ] work exists and author list matches
- [ ] year matches the version cited (preprint vs published differ)
- [ ] venue correct
- [ ] **the specific claim above is actually made by this work**

### Ye et al. 2024

**Claim as made in the draft:** ...at number is useful for ranking models, but it does not say where a call went wrong, and it usually reflects clean conditions in which the tools behave as expected. ### 2.3 Localizing the failure A smaller group of benchmarks breaks the call apart. RoTBench (Ye et al., 2024) evaluates tool selection, parameter identification, and content filling as separate stages, and does s...

- [ ] work exists and author list matches
- [ ] year matches the version cited (preprint vs published differ)
- [ ] venue correct
- [ ] **the specific claim above is actually made by this work**
