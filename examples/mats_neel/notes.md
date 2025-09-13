persona vectors: https://arxiv.org/pdf/2507.21509
dashboard: https://arxiv.org/html/2406.07882v3
emergent misalignment and persona: https://www.arxiv.org/pdf/2506.19823

1:39am started reading papers

2:02am running ideas:

user model impact on model behavior: 
- user is modeled as emotionally charged, how does this affect
    - benchmarks (tau bench, gsm8k, swebench?)
    - helpful/harmless (more likely to explain meth if user sad?)
    - model trying to steer the user (to feel less upset/etc)
- user is modeled as harmless/curious vs malicious, how does this affect
    - helpful/harmless/jailbreak success rate

emotionally charged models:
- model expresses some kind of 'distress' that we can see in activation space, how does this affect:
    - assistant benchmarks, helpful/harmless, etc

2:22am cleaning repository, starting to set up code

basic user modeling - prompt only
basic: prefix/suffix to system prompt/initial user message for single-turn benchmark
- annoyed/frustrated user
- happy user

medium: adjust system prompt/simulated user to something like tau-bench for multi-turn benchmark performance change

hard: same but for terminal bench (maybe even swe-bench :()

linear probes for user modeling
- does the llm model user emotion
- does it try to change it/how does behavior change with respect to that model

STOP 2:45am WED
START 12:10am THURS

start with most basic: 
user is emotionally charged from prompt suffix/prefix
run gsm8k with prompt suffix/prefix for frustrated user - vlliu80mt3stvpte3m
once this is launched, start planning linear probes for collecting activations/training

12:30am wrote design doc for writing code for the above

12:55am claude wrote code, going to test

1:07am fixed broken deps in deployment

1:27am unifying logging to make my life easier

2:12 improved logging + fixed vllm deps

2:30am STOP THURS

12:30am START FRI

3:47am may have overkilled but being able to distributedly run evals is probably important so its ok i think

5:00am STOP FRI
12:02pm START FRIDAY

gsm8k small sample size to confirm behavior
tau-bench small sample size to confirm behavior

run both again with nnsight/vllm to collect activations
run probes on activations

todo:

[ ] collect activations for probe training
[ ] this is a pain in the ass

[ ] figure out what analyze_results.py does
[ ] make collect results.py less pain in the ass
[ ] un-hardcode model choice
[ ] fix bifrost file copying to not be slow/sequential
[ ] make logigng/remote output look less bad



