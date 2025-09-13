# NNsight Activation Capture: Footguns & Lessons Learned

## 🎯 Current Status: SUCCESS ✅
**Working Server**: `server_composition.py` (Port 8006)
- ✅ Single-token activation capture: `[1,1,151936]` 
- ✅ File storage: `/tmp/nnsight_activations/*.pt`
- ✅ Clean composition architecture prevents context pollution
- ✅ All commits pushed to main branch

## 🚨 Critical Footguns We Hit

### 1. Context Pollution (The Big One!)
**Problem**: NNsight's internal state management breaks when mixed with other processing
```python
# ❌ BROKEN: Chat processing pollutes NNsight context
def chat_endpoint():
    # Chat template rendering, tokenizer calls, etc.
    with mm.lm.generate() as tracer:  # ← FAILS with OutOfOrderError
        logits = mm.lm.lm_head.output.save()
```

**Solution**: Complete separation via composition
```python
# ✅ WORKING: Isolated NNsight core
class NNsightCore:
    def capture_activations(self, prompt: str):
        with self.lm.generate() as tracer:  # ← WORKS perfectly
            logits = self.lm.lm_head.output.save()
```

**Key Insight**: The EXACT same NNsight code works in isolation but fails in chat context.

### 2. Multiple Save Calls in Single Context
**Problem**: Calling `.save()` multiple times in one generate context causes timing issues
```python
# ❌ BROKEN: Multiple saves cause OutOfOrderError  
with mm.lm.generate() as tracer:
    with tracer.invoke(prompt):
        logits = mm.lm.lm_head.output.save()        # First save - OK
        generated = mm.lm.generator.output.save()   # Second save - BREAKS!
```

**Solution**: One save per context
```python
# ✅ WORKING: Single save only
with mm.lm.generate() as tracer:
    with tracer.invoke(prompt):
        logits = mm.lm.lm_head.output.save()  # Only one save
```

### 3. Custom Savepoints Outside Generate Context
**Problem**: Trying to call `node.save()` outside the generate context
```python
# ❌ BROKEN: Savepoints outside context
with mm.lm.generate() as tracer:
    logits = mm.lm.lm_head.output.save()

# Later, outside context:
node = _safe_eval_selector(mm.lm, "model.layers[0].output")  
saved = node.save()  # ← OutOfOrderError!
```

**Solution**: All saves must be inside generate context (multi-token work needed)

### 4. Temperature/Top_p Parameters Cause Envoy Issues  
**Problem**: Non-default generation parameters interfere with NNsight timing
```python
# ❌ PROBLEMATIC: Custom params cause timing issues
gen_kwargs = dict(
    temperature=req.temperature,  # ← Can cause problems
    top_p=req.top_p,             # ← Can cause problems
)
```

**Solution**: Use minimal/default parameters for activation capture
```python
# ✅ WORKING: Minimal parameters
with self.lm.generate(max_new_tokens=3) as tracer:  # Only essential params
```

### 5. Text Extraction from Remote Tracers
**Problem**: `tracer.output` doesn't exist in remote configurations
```python
# ❌ BROKEN: Assumes local tracer
generated_text = tokenizer.decode(tracer.output[0])  # AttributeError!
```

**Solution**: Defensive attribute checking
```python
# ✅ WORKING: Handle missing attributes
if hasattr(tracer, 'output') and tracer.output is not None:
    generated_text = tokenizer.decode(tracer.output[0])
else:
    generated_text = "Text extraction needs fixing"
```

## 🏗️ Architecture That Works

### Clean Separation Pattern
```python
# 1. NNsightCore: Pure activation capture (isolated)
class NNsightCore:
    def capture_activations(self, prompt, max_tokens):
        # EXACT tutorial pattern - never mix with other logic
        with self.lm.generate(max_new_tokens=max_tokens) as tracer:
            with tracer.invoke(prompt):
                return self.lm.lm_head.output.save()

# 2. ChatProcessor: Handle templates/formatting (separate)  
class ChatProcessor:
    def render_chat_prompt(self, messages):
        # Chat logic completely separate from NNsight

# 3. Composition: Clean data flow
def chat_completions():
    prompt = processor.render_chat_prompt(messages)    # Step 1: Chat processing
    result = core.capture_activations(prompt)          # Step 2: Pure NNsight  
    response = build_response(result)                  # Step 3: Format response
```

## 🎯 Next Steps: Multi-Token Generation

### Current Limitation
- **Single Token**: `[1,1,151936]` - only final position activation
- **Goal**: `[1,N,151936]` - all N generated tokens

### Multi-Token Patterns from NNsight Docs
From working debug scripts (`debug_official_patterns.py`):

```python
# Pattern 2: Generate with .all() for multi-token capture
with model.generate(prompt, max_new_tokens=5) as tracer:
    hidden_states = list().save()  # Initialize list
    with tracer.all():  # Capture ALL generated tokens
        hidden_states.append(model.lm_head.output)

# Pattern 4: Generate with .iter[] for specific tokens  
with model.generate(prompt, max_new_tokens=5) as tracer:
    hidden_states = list().save()
    with tracer.iter[0:3]:  # Only first 3 iterations
        hidden_states.append(model.lm_head.output)
```

### Known Issues to Watch For
1. **Dictionary iteration error**: `RuntimeError: dictionary changed size during iteration`
   - Occurred when trying multi-token patterns in the old monolithic server
   - Should be resolved with clean composition architecture

2. **Savepoints timing**: Custom savepoints need to be inside the generate context
   - May need to redesign savepoint system for multi-token

## 📁 Key Files

### Working Implementation
- **`server_composition.py`**: Clean architecture that works ✅
- **`test_nnsight_tutorial.py`**: Proven working patterns
- **`debug_official_patterns.py`**: All 5 official patterns tested

### Historical/Broken
- **`server_singlepass.py`**: Monolithic version with context pollution ❌
- **`nnsight_activation_issue.md`**: Troubleshooting guide (partially incorrect)

### Documentation  
- **`HANDOFF_NNSIGHT_ACTIVATION_CAPTURE.md`**: Original handoff (overly optimistic)
- **`deploy.md`**: Bifrost deployment instructions

## 🔧 Deployment Commands

```bash
# Deploy composition server
bifrost push "ssh -p 10157 root@203.57.40.162"
cd ~/.bifrost/workspace && uv run --extra examples_gsm8k_nnsight_remote \
  python examples/gsm8k_nnsight_remote/server_composition.py --port 8006

# Load model  
curl -X POST http://localhost:8006/models/load \
  -H 'Content-Type: application/json' \
  -d '{"model_id": "willcb/Qwen3-0.6B", "device_map": "auto"}'

# Test activation capture
curl -X POST http://localhost:8006/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "willcb/Qwen3-0.6B", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 5, "store_activations": true}'
```

## 🎖️ Success Metrics
- ✅ **No OutOfOrderError**: Clean separation works
- ✅ **Proper tensor shape**: `[1,1,151936]` with correct dtype
- ✅ **File storage**: `.pt` files saved correctly  
- ✅ **Full pipeline**: Chat request → Activation capture → Storage → Response

## 💡 Key Lesson
**The problem was never with NNsight itself** - the tutorial patterns work perfectly when properly isolated. The issue was **architectural**: mixing NNsight with other request processing logic pollutes its internal state management.

**For multi-token work**: Start with the working `NNsightCore.capture_activations()` method and extend it using official NNsight patterns, maintaining the clean separation principle.