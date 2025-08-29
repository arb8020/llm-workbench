# Function Serialization for Rollouts

## Problem
In rollouts, we need to serialize functions (e.g., environment tool interactions) for storage, transmission, and later execution. This is critical for reproducibility and distributed evaluation.

## Approaches

### 1. Source Code as String
```python
import inspect
func_source = inspect.getsource(my_function)
```
**Pros:** Simple, human-readable  
**Cons:** Fragile with closures, imports, dependencies

### 2. Git Commit Hash + Function Path (RECOMMENDED)
```python
{
    "commit": "abc123def",
    "module": "rollouts.environments.calculator", 
    "function": "calculate_expression"
}
```
**Pros:** Reproducible, traceable, handles dependencies  
**Cons:** Requires git repo access during deserialization

### 3. Cloudpickle/Dill
```python
import cloudpickle
serialized = cloudpickle.dumps(my_function)
```
**Pros:** Handles closures, nested functions, most Python objects  
**Cons:** Version sensitivity, security concerns, debugging nightmare

### 4. Registry Pattern
```python
FUNCTION_REGISTRY = {
    "env_interact_v1": env_interact_function,
    "env_interact_v2": env_interact_function_v2
}
# Store just the key
```
**Pros:** Version control, safe deserialization  
**Cons:** Manual registration required

## Production Limitations of Cloudpickle

- **Version Brittleness:** Breaks across Python/library versions
- **Security Risk:** Can execute arbitrary code during deserialization  
- **Debugging:** Opaque binary blobs, impossible to inspect/diff
- **Dependency Hell:** Requires exact environment matching
- **Performance:** Large size, slow serialization

## Recommendation

Use **Git commit + function path** as primary approach:
- Human-readable serialization
- Perfect reproducibility via version control
- Easy debugging and diffing
- Security (just loading known code)
- Small storage footprint

Fallback to cloudpickle only for complex closures in development/experimentation.

## Implementation TODO

1. Create `FunctionSerializer` class in `rollouts/`
2. Implement git-based serialization method
3. Add function registry for common environment interactions
4. Update checkpoint system to use new serialization
5. Add tests for cross-version compatibility