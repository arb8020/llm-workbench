```python
prenorm_preactivation_BLD = layernorm(x_BLD) # record this
x_BLD = x_BLD + mha(prenorm_preactivation_BLD)
prenorm_preffn_BLD = layernorm(x_BLD) # record this
x_BLD = x_BLD + ffn(prenorm_preffn_BLD)
```

take above, check for magnitude >= 6.0


[x] get colab and broker working together
[x] fix some broker stuff
[x] fix deps
[x] collect activations from a MoE model (mistral/mixtral)
[x] collect activations from a push button script
[x] analyze activations according to the paper
[x] fix input data to be more robust (2048 tok, more sequences)
[x] add actual tokenizer

[x] get more models working
[x] wrote local model shape/layer name checker

[ ] todo: fix streaming logs from remote (tail -f not working asf)
[ ] consider reviving bifrost jobs

