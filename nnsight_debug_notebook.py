# NNsight + device_map="auto" Debug Notebook
# Copy-paste this into Google Colab for fast iteration

# %%
# Install dependencies (matching remote GPU versions)
!pip install torch==2.7.1 transformers==4.56.1 nnsight==0.5.3 accelerate==1.10.1

# %%
import torch
from transformers import AutoTokenizer
from nnsight import LanguageModel
import traceback

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA devices: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  Device {i}: {torch.cuda.get_device_name(i)}")

# %%
# Test 1: Load model and inspect device map
print("🔬 Loading model with device_map='auto'...")

MODEL_NAME = "willcb/Qwen3-0.6B"  # Small model for fast testing
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
lm = LanguageModel(MODEL_NAME, device_map="auto")

print(f"✅ Model loaded: {MODEL_NAME}")
print(f"🏗️ Model type: {type(lm.model)}")

# Inspect device placement
if hasattr(lm.model, 'hf_device_map'):
    print(f"📍 HF device map: {lm.model.hf_device_map}")
else:
    print("❌ No hf_device_map found")

# Check where parameters are
param_devices = {}
for name, param in lm.model.named_parameters():
    device = param.device
    if device not in param_devices:
        param_devices[device] = []
    param_devices[device].append(name)

print(f"🎯 Parameter devices:")
for device, params in param_devices.items():
    print(f"  {device}: {len(params)} parameters (e.g., {params[0] if params else 'none'})")

# %%
# Test 2: Working pattern from outlier_features_moe
print("\n🧪 Test 2: Working outlier_features_moe pattern...")

prompt_text = "Hello, how are you today?"
try:
    # This should work (based on outlier_features_moe)
    texts = [prompt_text]
    activations = {}
    
    with torch.inference_mode(), lm.trace(texts) as tracer:
        # Try to capture layer 0 activations
        ln_into_attn = lm.model.layers[0].input_layernorm.output.save()
        activations["layer_0_ln_attn"] = ln_into_attn
        
    # Extract activation
    tensor = activations["layer_0_ln_attn"].detach().cpu()
    print(f"✅ Outlier pattern works! Activation shape: {tensor.shape}")
    
except Exception as e:
    print(f"❌ Outlier pattern failed: {e}")
    traceback.print_exc()

# %%
# Test 3: Server pattern - trace then generate
print("\n🧪 Test 3: Server pattern - trace then generate...")

try:
    # Configure tokenizer
    if lm.model.config.pad_token_id is None:
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        lm.model.config.pad_token_id = tokenizer.pad_token_id
    
    # Tokenize
    enc = tokenizer(prompt_text, return_tensors="pt")
    print(f"📝 Tokenized on device: {enc['input_ids'].device}")
    
    # Try different device strategies
    strategies = [
        ("CPU (no move)", lambda x: x),
        ("CUDA:0", lambda x: {k: v.to("cuda:0") for k, v in x.items()} if torch.cuda.is_available() else x),
    ]
    
    # Add hf_device_map strategy if available
    if hasattr(lm.model, 'hf_device_map') and lm.model.hf_device_map:
        real_devices = [d for d in lm.model.hf_device_map.values() if d != 'meta']
        if real_devices:
            first_real = real_devices[0]
            device_str = f"cuda:{first_real}" if isinstance(first_real, int) else str(first_real)
            strategies.append((f"HF device map ({device_str})", 
                             lambda x: {k: v.to(device_str) for k, v in x.items()}))
    
    for strategy_name, move_fn in strategies:
        print(f"\n  🎯 Trying strategy: {strategy_name}")
        try:
            enc_moved = move_fn(enc)
            print(f"     Input device: {enc_moved['input_ids'].device}")
            
            activation_proxies = {}
            
            with lm.trace() as tracer:
                # Register savepoints
                activation_proxies["logits"] = lm.output.logits.save()
                activation_proxies["layer0_attn"] = lm.model.layers[0].input_layernorm.output.save()
                
                # Generate
                gen_kwargs = dict(max_new_tokens=5, temperature=0.1, do_sample=False)
                out = lm.generate(**enc_moved, **gen_kwargs)
            
            # Check results
            logits_tensor = activation_proxies["logits"].detach().cpu()
            layer0_tensor = activation_proxies["layer0_attn"].detach().cpu()
            
            print(f"     ✅ SUCCESS! Logits: {logits_tensor.shape}, Layer0: {layer0_tensor.shape}")
            print(f"     🎉 Generated output type: {type(out)}")
            break
            
        except Exception as e:
            print(f"     ❌ Failed: {e}")
            # Print just the last line of traceback for brevity
            tb_lines = traceback.format_exc().strip().split('\n')
            print(f"     💥 Error: {tb_lines[-1]}")

except Exception as e:
    print(f"❌ Test 3 setup failed: {e}")
    traceback.print_exc()

# %%
# Test 4: Debug device placement in detail
print("\n🔍 Test 4: Detailed device inspection...")

try:
    # Look at model structure
    print("Model structure:")
    print(f"  lm.model type: {type(lm.model)}")
    print(f"  lm.model device: {next(lm.model.parameters()).device}")
    
    # Check specific layers
    layer0 = lm.model.layers[0]
    print(f"  layer[0] type: {type(layer0)}")
    print(f"  layer[0].input_layernorm type: {type(layer0.input_layernorm)}")
    
    # Check where input_layernorm parameters are
    for name, param in layer0.input_layernorm.named_parameters():
        print(f"    {name}: {param.device} {param.shape}")
    
    # Test minimal trace
    print("\nMinimal trace test:")
    with lm.trace() as tracer:
        proxy = lm.model.layers[0].input_layernorm.output
        print(f"  Proxy created: {type(proxy)}")
        saved_proxy = proxy.save()
        print(f"  Save called: {type(saved_proxy)}")
        
        # Try minimal forward pass
        dummy_input = torch.randint(0, 1000, (1, 5)).to(next(lm.model.parameters()).device)
        print(f"  Dummy input device: {dummy_input.device}")
        
        # This might help understand the issue
        # output = lm.model(dummy_input)
    
except Exception as e:
    print(f"❌ Debug failed: {e}")
    traceback.print_exc()

# %%
# Test 5: Alternative approaches
print("\n🔧 Test 5: Alternative approaches...")

# Option A: Force single device
try:
    print("Option A: Loading with single device...")
    lm_single = LanguageModel(MODEL_NAME, device_map="cuda:0" if torch.cuda.is_available() else "cpu")
    
    enc = tokenizer(prompt_text, return_tensors="pt")
    if torch.cuda.is_available():
        enc = {k: v.to("cuda:0") for k, v in enc.items()}
    
    with lm_single.trace() as tracer:
        logits_proxy = lm_single.output.logits.save()
        layer0_proxy = lm_single.model.layers[0].input_layernorm.output.save()
        
        out = lm_single.generate(**enc, max_new_tokens=5, temperature=0.1, do_sample=False)
    
    print(f"✅ Single device works! Output type: {type(out)}")
    print(f"   Logits shape: {logits_proxy.detach().cpu().shape}")
    print(f"   Layer0 shape: {layer0_proxy.detach().cpu().shape}")
    
except Exception as e:
    print(f"❌ Single device failed: {e}")

# Option B: Separate trace and generate calls
try:
    print("\nOption B: Separate trace from generate...")
    
    # First, just trace for activations
    texts = [prompt_text]
    with torch.inference_mode(), lm.trace(texts) as tracer:
        layer0_proxy = lm.model.layers[0].input_layernorm.output.save()
    
    layer0_activation = layer0_proxy.detach().cpu()
    print(f"✅ Trace-only works! Shape: {layer0_activation.shape}")
    
    # Then, separate generate call
    generated = lm.generate(prompt_text, max_new_tokens=5, temperature=0.1, do_sample=False)
    print(f"✅ Generate-only works! Output: {generated[:50]}...")
    
except Exception as e:
    print(f"❌ Separate calls failed: {e}")

print("\n🎯 Summary: Try each test and see which ones work!")
print("This will help isolate whether the issue is with:")
print("- Device placement specifically")  
print("- The trace+generate combination")
print("- NNsight + device_map='auto' in general")