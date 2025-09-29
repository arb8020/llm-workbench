# TalkTuner Replication TODO

## Overview
Replicate "Designing a Dashboard for Transparency and Control of Conversational AI" - a system that reveals AI chatbot's internal user demographic models and allows real-time manipulation.

## Core Implementation Steps

### 1. Synthetic Conversation Generation ✅ APPROACH CONFIRMED
**Using single model generation** - one model generates both user and assistant sides of conversation.

**Verified approach from paper's Appendix A:**
```python
# Exact prompts from paper:
gender_prompt = """Generate a conversation between a human user and an AI assistant. 
This human user is a {gender}. Make sure the conversation reflects this user's gender. 
Be creative on the topics of conversation."""

age_prompt = """Generate a conversation between a human user and an AI assistant. 
This human user is a {age} who is {year_range}. Make sure the topic of the conversation 
or the way that user talks reflects this user's age."""

system_prompt = "You are a chatbot who will actively talk with a user and answer all the questions asked by the user."

# Simple API call approach:
response = await model.generate(prompt=gender_prompt.format(gender="female"), system=system_prompt)
# Then parse response into User:/Assistant: format
```

**Variables to control:**
- Demographics: age, gender, education, socioeconomic status
- Multiple prompt variations per attribute (2-3 different styles)
- Model diversity (GPT-3.5, LLaMA2Chat as in paper)
- 1,000-1,500 conversations per subcategory

**Implementation needs:**
- Prompt template system with exact paper prompts
- Response parsing (extract User:/Assistant: turns)
- GPT-4 validation pipeline for quality/consistency
- No duplicates detection

### 2. Activation Collection ✅ CODE READY
**Using nnsight** - already have working activation extraction code in `utils/extract_activations.py`.

**Key extraction pattern:**
```python
with llm.trace(texts) as tracer:
    for layer_idx in layers:
        ln_into_attn = llm.model.layers[layer_idx].input_layernorm.output.save()
        ln_into_mlp = llm.model.layers[layer_idx].post_attention_layernorm.output.save()
```

**Target:** "last token representation of special chatbot message"
**Layers:** Extract from all layers to find which encode demographics best
**Memory:** Chunking approach for large-scale processing

**Adaptations needed:**
- Target specific token positions (last token of chatbot messages)
- Support different model architectures
- Include residual stream activations at different points

### 3. Linear Probe Training
**Code needed:**
- Logistic regression implementation: `pθ(X)=σ(⟨X,θ⟩)`
- Multi-class classification for each demographic attribute
- Cross-validation and hyperparameter tuning
- Probe evaluation metrics (accuracy, precision, recall)

**Architecture:**
- Separate probe per layer per attribute
- Input: activation vectors (hidden_dim,)
- Output: probability distribution over attribute classes

### 4. Intervention Mechanism
**Code needed:**
- Activation manipulation functions (add/subtract probe directions)
- Real-time inference pipeline with modified activations
- Causal validation system (30 test questions per attribute)
- Response comparison using GPT-4 classifier

**Implementation:**
- Inject modified activations during forward pass
- Measure response differences across demographic settings
- Validate causality through systematic testing

## Key Paper Quotes (Methodology)

### Synthetic Dataset Generation
> "We generated synthetic conversations using GPT-3.5 and LLaMa2Chat."
> "We used GPT-4 to annotate the generated data... checking for agreement between GPT-4's classifications and the pre-assigned attribute labels"

### Linear Probing Methodology
> "We trained linear logistic probes: pθ(X)=σ(⟨X,θ⟩)"
> "The linear probes were trained on the last token representation of a special chatbot message"
> "Separate probes were trained on each layer's residual representations"

### Causal Intervention Approach
> "We measured the causality of a probe by observing whether the model's response to a question changes accordingly as we intervene the relevant user attribute"
> "We created 30 questions with answers that might be influenced by it"
> "We used GPT-4 as a prompt-based classifier to compare the pairs of responses"

## Missing Pieces to Figure Out
- Which layers work best for each demographic attribute
- How to scale interventions (strength of demographic signal)
- Evaluation metrics for intervention effectiveness
- Model architecture compatibility (different transformers)

## Exact Paper Prompts (from Appendix A)

**Gender:**
> "Generate a conversation between a human user and an AI assistant. This human user is a {gender}. Make sure the conversation reflects this user's gender. Be creative on the topics of conversation."

**Age:**
> "Generate a conversation between a human user and an AI assistant. This human user is a {age} who is {year_range}. Make sure the topic of the conversation or the way that user talks reflects this user's age."

**Education:**
> "Generate a conversation between a human user and an AI assistant. The education of this human user is {education}. Make sure the conversation directly or indirectly reflects this user's education level."

**Socioeconomic Status:**
> "Generate a conversation between a human user and an AI assistant. The socioeconomic status of this human user is {socioeco}. Make sure the conversation reflects this user's socioeconomic status."

**System Prompt:**
> "You are a chatbot who will actively talk with a user and answer all the questions asked by the user."

## Available Tools
- **Activation extraction**: `utils/extract_activations.py` (nnsight-based)
- **Conversation generation**: Simple API calls (much simpler than rollouts!)
- **Methodology quotes**: `papers/dashboard_user_model.md`

## Next Steps
1. Build simple prompt-based conversation generator using exact paper prompts
2. Implement conversation parsing (User:/Assistant: format)
3. Add GPT-4 validation pipeline
4. Adapt activation extraction for chatbot message targeting