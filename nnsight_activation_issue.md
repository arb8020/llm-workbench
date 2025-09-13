Short version
- Don’t pass the prompt when you open generate; arm all .save() hooks first, then start the run with tracer.invoke(...). This guarantees no early-layer compute happens before your registrations. ([nnsight.net](https://nnsight.net/_modules/nnsight/models/NNsightModel/?utm_source=openai))
- For device_map="auto", let NNsight/Accelerate dispatch the model; keep inputs on CPU and avoid moving anything to "meta". If you need multi‑GPU sharding, construct LanguageModel(..., device_map="auto", dispatch=True). ([nnsight.net](https://nnsight.net/_modules/nnsight/models/NNsightModel/?utm_source=openai))
- Read tensors with proxy.value (NNsight proxy API), not detach(). ([github.com](https://github.com/ndif-team/nnsight/blob/main/README.md?utm_source=openai))

Minimal fix (drop the inner trace-then-generate; use generate + invoke)
```python
gen_kwargs = dict(
    max_new_tokens=req.max_tokens,
    temperature=max(req.temperature, 1e-5),
    top_p=req.top_p,
    do_sample=req.temperature > 0,
)

activation_proxies = {}

with mm.lock:
    # single-pass run with hooks armed before any compute
    with mm.lm.generate(**gen_kwargs) as tracer:
        # register savepoints first
        try:
            # logits via lm_head; or use mm.lm.output.save() if you want the full HF output dict
            activation_proxies["_logits"] = mm.lm.lm_head.output.save()
        except Exception:
            pass
        for sp in mm.savepoints:
            try:
                node = _safe_eval_selector(mm.lm, sp.selector)
                activation_proxies[sp.name] = node.save()
            except Exception as e:
                activation_proxies[sp.name] = {"error": f"Could not save '{sp.selector}': {e}"}

        # now start execution; pass either the raw prompt or a tokenized dict (on CPU)
        with tracer.invoke(prompt_text):
            pass    # nothing to do; run happens on context exit

# get text from generated ids if you need it
gen_ids = tracer.output              # token ids from GenerationMixin.generate
reply_text = mm.tokenizer.decode(gen_ids[0])

# materialize saved activations
for k, proxy in activation_proxies.items():
    if isinstance(proxy, dict) and "error" in proxy:
        continue
    try:
        tensor = proxy.value         # NNsight proxy -> torch.Tensor
        # save tensor to disk...
    except Exception as e:
        # handle errors...
        pass
```

Why this works (with docs/source)
- Using tracer.invoke when no inputs were passed to .trace()/.generate defers creation of the invoker until after your hooks are armed. That’s the supported way to register first, then run. ([nnsight.net](https://nnsight.net/_modules/nnsight/models/NNsightModel/?utm_source=openai))
- LanguageModel accepts strings, token IDs, or a dict of tensors; inputs are preprocessed by the Invoker and batched, and arguments are forwarded to transformers’ GenerationMixin.generate. So you can pass the raw prompt, or a CPU tokenized dict, and leave device placement to Accelerate. ([nnsight.net](https://nnsight.net/_modules/nnsight/modeling/language/?utm_source=openai))
- After a tracing/generation context exits, outputs and saved proxies are available; use proxy.value to retrieve tensors. ([github.com](https://github.com/ndif-team/nnsight/blob/main/README.md?utm_source=openai))
- For sharded/multi‑GPU loads, NNsight’s loader uses Accelerate’s load_checkpoint_and_dispatch when dispatched; avoid pushing inputs to a ‘meta’ device yourself. If you hit “meta” errors with auto, prefer dispatch=True at model load. ([nnsight.net](https://nnsight.net/_modules/nnsight/models/NNsightModel/?utm_source=openai))

Optional tweaks
- If you specifically want the full HF CausalLM output (e.g., logits dict) in one go, you can save mm.lm.output and then index "logits" from the returned dict. ([nnsight.net](https://nnsight.net/notebooks/tutorials/get_started/start_remote_access/?utm_source=openai))
- To capture across generated time steps, call invoker.next() between saves. ([github.com](https://github.com/ndif-team/nnsight?utm_source=openai))

References
- README examples: trace vs generate, invoke usage, and post-context retrieval. ([github.com](https://github.com/ndif-team/nnsight?utm_source=openai))
- API docs: LanguageModel input types; Tracer/Invoker semantics. ([nnsight.net](https://nnsight.net/documentation/models/?utm_source=openai))
- Tutorials: Cross‑prompt interventions; logit lens patterns. ([nnsight.net](https://nnsight.net/notebooks/features/cross_prompt/?utm_source=openai))
- Issues noting meta/auto/dispatch patterns. ([github.com](https://github.com/ndif-team/nnsight/issues/83?utm_source=openai))
