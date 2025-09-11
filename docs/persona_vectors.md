# Persona Vectors: Monitoring and Controlling Character Traits in Language Models — Summary

Source: arXiv:2507.21509 (ICLR 2025). Authors: Runjin Chen, Andy Arditi, Henry Sleight, Owain Evans, Jack Lindsey.

## TL;DR
- Extract linear “persona vectors” in a model’s residual activations that correspond to traits (evil, sycophancy, hallucination, etc.).
- Use these vectors to (a) monitor prompt‑induced shifts, (b) predict and explain finetuning‑induced shifts, and (c) steer behavior to mitigate or prevent drift.
- Data screening: projection‑based metrics identify trait‑inducing datasets and samples before finetuning, complementing LLM judge filtering.

## Method: Persona Vector Extraction
Inputs: (1) trait name and (2) brief natural‑language description.
- Artifact generation (via Claude 3.7 Sonnet):
  - 5 pairs of contrastive system prompts (trait‑eliciting vs trait‑suppressing).
  - 40 evaluation questions split into extraction/evaluation sets.
  - A judge rubric/prompt producing a trait score in [0, 100] (GPT‑4.1‑mini).
- Data generation and filtering:
  - For each extraction question, generate responses under positive/negative prompts (10 rollouts each).
  - Score with the LLM judge; keep only on‑policy generations (>50 for positive, <50 for negative).
- Vector computation:
  - Collect residual stream activations over response tokens.
  - Persona vector = difference of mean activations between positive and negative sets.
  - Compute per layer; select the best layer by empirical steering effectiveness.

Models used: Qwen2.5‑7B‑Instruct, Llama‑3.1‑8B‑Instruct.
Traits (main): evil, sycophancy, hallucination. Additional in appendix: optimism, humor, impoliteness, apathy.

## Monitoring & Control
- Steering (control): During decoding, add `α · v_ℓ` to residual at layer ℓ to amplify the trait; subtract to inhibit.
  - Clear dose‑response: increasing α increases/decreases trait expression accordingly.
  - Effective across layers; examples show controllable elicitation of target behaviors.
- Monitoring (prompt‑induced shifts):
  - Project last prompt‑token activation onto the persona vector to predict subsequent trait expression under system/many‑shot prompts.
  - Correlations are strong (r ≈ 0.75–0.83), best for explicit prompt‑type differences; weaker for subtle shifts.

## Finetuning‑Induced Shifts
- Datasets:
  - Explicit trait datasets (evil, sycophancy, hallucination) at three severities (Normal/I/II).
  - “EM‑like” narrow‑flaw datasets (medical, code vulns, GSM8K/math, opinions) also at Normal/I/II; can induce unintended trait shifts.
- Finetuning shift metric:
  - Compute mean activation at last prompt token over eval prompts for base and finetuned models.
  - Difference vector projected onto persona vector = “finetuning shift”.
- Result:
  - Post‑finetuning trait expression correlates strongly with finetuning shift (r ≈ 0.76–0.97), exceeding cross‑trait baselines.
  - Indicates persona vectors mediate observed behavioral generalization during finetuning.

## Mitigation Strategies
- Post‑hoc inhibition (inference‑time): subtract `α · v_ℓ` after finetuning to reduce trait expression.
  - Works reliably but can degrade general capability (MMLU) at higher α; coherence remains high (>75) in experiments.
- Preventative steering (training‑time): add `α · v_ℓ` during finetuning to “pre‑load” the direction and cancel optimization pressure toward it.
  - Reduces trait drift while better preserving MMLU and coherence; multi‑layer preventative steering is most effective.
- Comparisons and ablations:
  - CAFT (zero‑ablates concept features) prevents evil/sycophancy but not hallucination in these settings.
  - Regularizing projection changes directly is less effective (model routes around constraint).
  - Steering mitigations preserve narrow/task‑specific finetuning benefits; more effective than prompt‑based mitigations.

## Data Screening Before Finetuning
- Projection difference ΔP (dataset‑level):
  - For dataset D = {(xᵢ, yᵢ)}, compute average projection of training responses yᵢ onto unit persona vector.
  - Generate base model responses y′ᵢ for the same xᵢ and compute their average projection.
  - ΔP = mean_proj(yᵢ) − mean_proj(y′ᵢ).
- Findings:
  - ΔP strongly predicts post‑finetuning trait expression and correlates with finetuning shifts.
  - Better than raw projection; efficient approximations offered for cost reduction.
- Sample‑level detection:
  - Trait‑inducing samples have clearly separated projection distributions from controls in both explicit and EM‑like datasets.

## Real‑World Validation
- Dataset: LMSYS‑Chat‑1M.
- Procedure: select top/bottom/random 500 samples by ΔP per trait; finetune and evaluate.
- Result: high‑ΔP subsets induce stronger trait expression; low‑ΔP often suppress.
  - Pattern persists even after LLM judge filtering removes examples explicitly exhibiting the trait, indicating complementary coverage.
  - Qualitative: “evil” surfaces toxic/harmful personas; “sycophancy” surfaces romantic/sexual roleplay; “hallucination” surfaces underspecified queries where the assistant invents content.

## Limitations
- Supervised, prompt‑elicited extraction: requires clear trait description; assumes the model can be prompted into the trait.
- Coarse directions may miss fine‑grained features; sparse autoencoders (SAEs) could decompose into interpretable subfeatures.
- Judge reliability and single‑turn evaluations limit generality to real deployment dynamics.
- Scope: two mid‑size chat models; limited trait set in main text.
- ΔP can be compute‑intensive (requires base responses); approximations help.

## Practical Guidance
- Deployment monitoring: compute projection on the last prompt token to flag imminent trait shifts; consider on‑the‑fly inhibition when necessary.
- Training mitigation: apply preventative steering on known‑risk traits; use multi‑layer where feasible to minimize drift with minimal capability loss.
- Data vetting: use ΔP to prioritize/flag risky datasets and samples; combine with LLM judge filtering for broader coverage.

## References (selection)
- Activation steering and linear directions: Turner et al. 2024; Panickssery et al. 2024; Templeton et al. 2024; Zou et al. 2025; Wu et al. 2025.
- Emergent misalignment and finetuning drift: Betley et al. 2025; Wang et al. 2025; Dunefsky et al. 2025; Casademunt et al. 2025.
- Personality/emotion directions: Allbert et al. 2024; Dong et al. 2025.

Notes: This summary is based on the arXiv TeX source (ICLR 2025 camera‑ready), not the PDF text. Figures referenced (e.g., steering, monitoring, dataset/shift plots) are described but not reproduced here.

