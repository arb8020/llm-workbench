# Setup Instructions for Llama-Stack Integration

## User Setup Required

Before the next Claude session can validate the JAX implementation, you need to:

### 1. Get Llama Model Access
- ✅ **DONE** - You already have access (see `llama.txt`)
- Valid for 48 hours, 5 downloads per model

### 2. Download Llama-3.2-1B-Instruct Model

Run these commands on your GPU instance:

```bash
# Install llama-stack
pip install llama-stack -U

# Check available models
llama model list

# Download the model (use the URL from llama.txt when prompted)
llama model download --source meta --model-id Llama-3.2-1B-Instruct
```

**Important**: When prompted for the custom URL, paste this:
```
https://llama3-2-lightweight.llamameta.net/*?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiNGJ1MW55cHh5c2dnaGY1a2I0dWUydTQ0IiwiUmVzb3VyY2UiOiJodHRwczpcL1wvbGxhbWEzLTItbGlnaHR3ZWlnaHQubGxhbWFtZXRhLm5ldFwvKiIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc1NzE4ODI2Mn19fV19&Signature=Ei5q4vmy7zA-k-unLBkH2T6Gc4Fj0vVlOEEIYfq45qnIQASVf-X01mtIgvuHFlY%7EMCZ0ALeqnsBJPXas-aGIIbFX5wAdxAvwuY9xtqdwgu0iRpMCD1pEQnF1PzD5WbzktfnWBn-D81CpystNq1YXaCTGEUtQ32F-RB4qsZxFVyZYkhY1KSMXT4z0eiiWszW9DtYef4dZus2oFlIHr6HF5i6uG0wNLhJYMHxkbbFSYY%7Ero4P6uqCh39%7EgVUj3jIHUeY1%7EfXNqyoyVqxc4i6vUKTklbWcVHqCXYr%7Eqk2yui-LqGgiCFWWZ14QPDdbIBr7HGOAlbygL0hhU-n5miNRSUQ__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=797590686546927
```

### 3. Verify Installation

```bash
# Check if model downloaded successfully
llama model list --show-downloaded

# Test basic model loading (if supported)
python -c "import llama_stack; print('✅ llama-stack ready')"
```

### 4. For Next Claude Session

Once the model is downloaded, the next Claude can run:

```bash
# Test current faithful implementation architecture
broker instances list --name "llama3-jax" --ssh-only | xargs -I {} bifrost deploy {} 'uv run python engine/scripts/dev/llama3_jax/compare_faithful.py --mode faithful --tokens 5'

# Then update and test with real Llama model
# (Implementation will be done by next Claude)
```

## Alternative: Manual Download

If llama-stack doesn't work, you can also:
1. Visit https://www.llama.com/llama-downloads/ 
2. Download Llama-3.2-1B-Instruct manually
3. Convert to format compatible with our JAX implementation

## Expected File Structure After Download

```
~/.llama/checkpoints/Llama-3.2-1B-Instruct/
├── model.pth              # Model weights
├── params.json            # Model configuration  
├── tokenizer.model        # Tokenizer
└── ...                    # Other files
```

## Next Steps for Claude

1. **Verify model downloaded successfully**
2. **Update weight loading in `solution_entropix_faithful.py`**
3. **Test JAX implementation with real weights**
4. **Compare against local Llama reference**
5. **Validate multi-token generation and KV cache**

The faithful JAX implementation is ready - just needs real weights!