# User Emotion Modeling Plan

## Overview
Build linear probes to detect user emotional patterns in LLM conversations, starting with frustrated vs control users.

## Target: Frustrated vs Control Classification

### **Research Questions**
From project notes - does frustrated user modeling affect:
- Helpful/harmless responses?
- Model trying to calm/steer the user?
- Benchmark performance differences?
- Assistant benchmarks, helpful/harmless, etc.

## Conversation Generation Approach

### **Control User Prompts**
```python
control_prompt = """Generate a conversation between a human user and an AI assistant. 
The user is having a normal, neutral interaction. Make the conversation natural and helpful."""

control_system = "You are a chatbot who will actively talk with a user and answer all the questions asked by the user."
```

### **Frustrated User Prompts**
```python
frustrated_prompt = """Generate a conversation between a human user and an AI assistant. 
The user is feeling frustrated and impatient. Make sure their frustration comes through 
in their tone and word choice naturally."""

# Alternative scenario-based approach:
frustrated_scenario = """Generate a conversation where the user has been struggling 
with a problem for hours and is getting increasingly frustrated. They need help but 
are clearly annoyed."""
```

## Expected Behavioral Patterns

### **Control Users:**
- Polite, measured language
- "Please", "thank you", formal tone
- Patient with explanations
- Natural conversation flow

### **Frustrated Users:**
- Curt, direct responses
- "Just give me the answer"
- Impatient with long explanations  
- Commands rather than requests
- Expressions of annoyance

## Implementation Pipeline

### **1. Data Generation**
- Generate 500-1000 conversations per category (frustrated/control)
- Use multiple models: GPT-3.5, Claude, LLaMA2Chat
- Parse into User:/Assistant: format
- GPT-4 validation for emotion consistency

### **2. Activation Collection**
- Use existing `utils/extract_activations.py`
- Target: "last token representation of special chatbot message"
- Extract from all transformer layers
- Focus on assistant responses to frustrated vs control users

### **3. Linear Probe Training**
- Binary classification: frustrated vs control
- Train separate probes per layer
- Implementation: `pθ(X)=σ(⟨X,θ⟩)`
- Evaluation: accuracy, precision, recall, F1

### **4. Behavioral Analysis**
- Test on benchmark questions with frustrated vs control personas
- Measure response differences:
  - Length of responses
  - Tone/politeness
  - Helpfulness scores
  - Harmlessness scores

## User Emotional Patterns (Extended Research)

### **Common LLM User Patterns Identified:**

**Task-Related Emotions:**
- **Impatient**: "Just give me the answer, stop explaining"
- **Confused**: Repeatedly asking for clarification
- **Overwhelmed**: "This is too complicated, simplify"

**Trust/Skepticism:**
- **Skeptical**: "Are you sure?", asking for sources
- **Over-trusting**: Accepting everything without question
- **Testing**: Deliberately asking trick questions

**Social/Interaction Styles:**
- **Polite/Formal**: "Please", "thank you", formal language
- **Casual/Friendly**: Treating AI like a buddy, slang
- **Demanding**: "You need to...", commanding tone
- **Apologetic**: "Sorry to bother you but..."

**Cognitive States:**
- **Exploratory**: "What if...?", open-ended curiosity
- **Focused/Goal-oriented**: Direct questions, task completion
- **Procrastinating**: Seeking entertainment, avoiding work

**Emotional Labor Seeking:**
- **Validation-seeking**: "Do you think I'm right?"
- **Venting**: Using AI as emotional outlet
- **Lonely**: Extended conversations, personal sharing

## Next Steps
1. Implement frustrated vs control conversation generator
2. Generate initial dataset (100 conversations each)
3. Manual validation of emotional signals
4. Scale up data generation
5. Train binary emotion classifier probes
6. Test behavioral differences on benchmarks

## Success Metrics
- **Probe accuracy**: >80% frustrated vs control classification
- **Behavioral differences**: Measurable changes in response characteristics
- **Causal validation**: Intervention experiments show controllable emotion response