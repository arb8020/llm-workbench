#!/bin/bash
# Pre-deployment validation script
# Run this before launching the experiment to catch issues early

echo "🔍 Pre-deployment validation checks..."
echo

# Check 1: Python syntax validation
echo "1️⃣ Checking Python syntax..."
python -m py_compile worker_experiment.py || exit 1
python -m py_compile launch_experiment.py || exit 1
echo "✅ Syntax check passed"
echo

# Check 2: Import validation
echo "2️⃣ Checking imports..."
python -c "
import sys
try:
    from worker_experiment import process_job, Job
    from launch_experiment import PROMPT_VARIANTS
    from rollouts.evaluation import evaluate_sample, EvalSample
    from rollouts.dtypes import Message, Endpoint, AgentState, Trajectory
    print('✅ All imports successful')
except ImportError as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)
" || exit 1
echo

# Check 3: Core data structures
echo "3️⃣ Checking data structure compatibility..."
python -c "
import sys
try:
    from rollouts.evaluation import EvalSample
    from rollouts.dtypes import Message, Trajectory
    
    # Verify EvalSample has agent_states (not agent_state)
    import inspect
    fields = inspect.signature(EvalSample.__init__).parameters
    if 'agent_states' not in fields:
        print('❌ EvalSample missing agent_states field')
        sys.exit(1)
    
    # Verify Message has to_json method
    msg = Message(role='user', content='test')
    if not hasattr(msg, 'to_json'):
        print('❌ Message missing to_json method')
        sys.exit(1)
    
    print('✅ Data structures compatible')
except Exception as e:
    print(f'❌ Data structure error: {e}')
    sys.exit(1)
" || exit 1
echo

# Check 4: Run local tests
echo "4️⃣ Running local serialization tests..."
python test_worker_locally.py || exit 1
echo

# Check 5: Verify experiment configuration
echo "5️⃣ Checking experiment configuration..."
python -c "
import sys
try:
    from launch_experiment import PROMPT_VARIANTS
    
    required_variants = ['control', 'frustration', 'impatience', 'anxiety', 'collaborative', 'patience', 'calm']
    for variant in required_variants:
        if variant not in PROMPT_VARIANTS:
            print(f'❌ Missing variant: {variant}')
            sys.exit(1)
    
    print(f'✅ All {len(required_variants)} variants configured')
except Exception as e:
    print(f'❌ Configuration error: {e}')
    sys.exit(1)
" || exit 1
echo

echo "🎉 All pre-deployment checks passed!"
echo "🚀 Ready to run: python launch_experiment.py --experiment-name YOUR_NAME --samples N"
echo