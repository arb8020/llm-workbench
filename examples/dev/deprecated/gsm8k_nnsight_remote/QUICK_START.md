# NNsight Activation Capture - Quick Start

## 🎯 What We Built

**Working NNsight activation capture server** with clean composition architecture.
- ✅ **Single-token capture**: Shape `[1,1,151936]` 
- ✅ **File storage**: Activations saved to `.pt` files
- ✅ **No context pollution**: Clean separation prevents OutOfOrderError
- 🎯 **Ready for multi-token extension**

## 🚀 Quick Test (Next Maintainer Start Here!)

### 1. Test the Current Working Server
```bash
# Test our working server (should be running on port 8006)
cd examples/gsm8k_nnsight_remote
python test_composition_server.py --host 203.57.40.162 --port 8006
```

**Expected Output:**
```
🧪 Running NNsight Composition Server Test Suite
🏥 Testing server health...
   ✅ Server healthy: {'status': 'ok', 'loaded_models': ['willcb/Qwen3-0.6B']}
📦 Testing model loading: willcb/Qwen3-0.6B...
   ✅ Model loaded: {'ok': True, 'model': 'willcb/Qwen3-0.6B'}
🧠 Testing pure NNsight core...
   ✅ Pure NNsight core works!
      Shape: [1, 1, 151936]
💬 Testing basic chat completion...
   ✅ Basic chat completion works!
🎯 Testing activation capture (MAIN FEATURE)...
   ✅ Activation capture works!
      _logits: shape=[1,1,151936], dtype=torch.float32, size=0.58MB
      📝 CURRENT: Single-token capture

✅ Basic Tests: 5/5 passed
📝 ACTIVATION CAPTURE: Single-token working (ready for multi-token extension)

📋 WHAT'S NEXT:
   🔧 TODO: Extend to multi-token capture
   📚 See: debug_official_patterns.py for tracer.all() patterns
   🎯 Goal: Change shape from [1,1,151936] to [1,N,151936]
```

### 2. Deploy Your Own Server (Optional)
```bash
# Deploy fresh server for testing
python deploy_and_smoke_test.py --model willcb/Qwen3-0.6B --port 8007

# Test your server
python test_composition_server.py --host YOUR_SERVER_IP --port 8007
```

## 📁 Key Files (What to Use)

### ✅ **Production Files** (Use These)
```bash
server_composition.py           # MAIN - Working composition server ⭐
debug_official_patterns.py     # Reference patterns for multi-token work
test_composition_server.py     # Test client (validates what works)
test_nnsight_tutorial.py       # Basic NNsight patterns that work
deploy_and_smoke_test.py       # Easy deployment + testing
```

### 📚 **Documentation** (Read These)
```bash
../../NNSIGHT_FOOTGUNS_AND_LESSONS.md    # ⭐ CRITICAL debugging lessons
README.md                                # Comprehensive documentation  
../../CLEANUP_PLAN.md                   # File organization guide
```

### 🗂️ **Legacy/Reference** (Can Ignore)
```bash
server_singlepass.py           # Old version with context pollution
deploy_and_evaluate.py         # Full GSM8K evaluation (more complex)
quick_test.py                  # Basic server test
test_server_pushbutton.py      # Legacy test
```

## 🎯 Multi-Token Development (Next Steps)

### Current Status
- **Shape**: `[1,1,151936]` - only final token activation
- **Goal**: `[1,N,151936]` - all N generated tokens

### Starting Point
1. **File**: `server_composition.py`
2. **Method**: `NNsightCore.capture_activations()`
3. **Current pattern**:
   ```python
   with self.lm.generate(max_new_tokens=max_new_tokens) as tracer:
       with tracer.invoke(prompt):
           logits = self.lm.lm_head.output.save()  # Single token
   ```

### Multi-Token Patterns (From debug_official_patterns.py)
```python
# Pattern: tracer.all() for all tokens
with model.generate(prompt, max_new_tokens=5) as tracer:
    hidden_states = list().save()  # Initialize list
    with tracer.all():  # Capture ALL generated tokens
        hidden_states.append(model.lm_head.output)
```

### Validation
Use `test_composition_server.py` to verify multi-token shapes:
- **Success**: Shape changes to `[1,5,151936]` for 5 tokens
- **Test detects**: Multi-token automatically and shows "🎉 BREAKTHROUGH: Multi-token capture!"

## 🏗️ Architecture (Why It Works)

### The Breakthrough: Composition
```python
# ❌ OLD: Context pollution (failed)
def chat_endpoint():
    # Chat processing + NNsight mixed together
    with model.generate() as tracer:  # ← Breaks with OutOfOrderError
        
# ✅ NEW: Clean separation (works)
class NNsightCore:           # Pure NNsight operations (isolated)
class ChatProcessor:         # Chat templates (separate)
class ActivationStorage:     # File storage (separate)

# Clean flow: chat → nnsight → storage → response
```

### Key Insight
**The problem was never with NNsight itself** - tutorial patterns work perfectly when properly isolated. The issue was mixing NNsight with chat request processing.

## 🚨 Common Pitfalls (Already Solved)

1. **Context Pollution**: ✅ Fixed with composition architecture
2. **Multiple Saves**: ✅ Only one `.save()` per generate context  
3. **Timing Issues**: ✅ Minimal parameters, no temperature/top_p conflicts
4. **Text Extraction**: ✅ Handles missing `tracer.output` gracefully

See `NNSIGHT_FOOTGUNS_AND_LESSONS.md` for complete debugging history.

## 🎉 Success Metrics

- ✅ **No OutOfOrderError**: Clean separation works
- ✅ **Proper tensor shape**: `[1,1,151936]` with correct dtype
- ✅ **File storage**: `.pt` files saved correctly
- ✅ **Full pipeline**: Chat request → Activation capture → Storage → Response
- 🎯 **Ready for multi-token**: Solid foundation established

**Next step**: Extend `capture_activations()` using official patterns! 🚀