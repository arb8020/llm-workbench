Variable names should be concise and informative. For a tensor, nothing is more informative than how many dimensions it has, and what those dimensions represent.

Get Noam Shazeer’s stories in your inbox
Join Medium for free to get updates from this writer.

Enter your email
Subscribe
We have been keeping this convention at Character.AI since 2022. Give it a try and let me know if you feel saner:

Designate a system of single-letter names for logical dimensions, e.g. B for batch size, L for sequence length, etc., and document it somewhere in your file/project/codebase
When known, the name of a tensor should end in a dimension-suffix composed of those letters, e.g. input_token_id_BL for a two-dimensional tensor with batch and length dimensions.
That’s all. You can use shape suffixes with torch, JAX, whatever. See the example below.
""" Example Transformer code with shape suffixes.

This code is incomplete and possibly has bugs.  Don't try to run it.
Its purpose is to illustrate shape suffixes.

Dimension key:

B: batch size
L: sequence length
M: memory length (length of sequence being attended to)
D: model dimension (sometimes called d_model or embedding_dim)
V: vocabulary size
F: feed-forward subnetwork hidden size
H: number of attention heads in a layer
K: size of each attention key or value (sometimes called d_kv)
"""

def transformer(input_token_id_BL, params):
  hidden_BLD = params.embedding_VD[input_token_id_BL]
  for layer_num in range(params.num_layers):
     hidden_BLD += attention(hiddden_BLD, params.attention_params[i])
     hidden_BLD += ffn(hiddden_BLD, params.ffn_params[i])
  hidden_BLD = layer_norm(hidden_BLD, params.final_layernorm_params)
  logits_BLV = torch.matmul(hidden_BLD, params.embedding_VD.T)
  return logits_BLV

def ffn(input_BLD, params):
   input_BLD = layer_norm(input_BLD, params.layernorm_params)
   hidden_BLF = torch.gelu(torch.matmul(input_BLD, params.w_in_DF))
   output_BLD = torch.matmul(hidden_BLF, params.w_out_FD)
   return output_BLD

def attention(input_BLD, params):
   input_BLD = layer_norm(input_BLD, params.layernorm_params)
   query_BLHK = torch.einsum('BLD,DHK->BLHK', input_BLD, params.w_q_DHK)
   key_BMHK = torch.einsum('BLD,DHK->BLHK', input_BLD, params.w_k_DHK)
   value_BMHK = torch.einsum('BLD,DHK->BLHK', input_BLD, params.w_k_DHK)
   logits_BHLM = torch.einsum('BLHK,BMHK->BHLM', query_BLHK, key_BMHK)
   B, L, H, K = query_BLHK.shape()
   logits_BHLM /= K ** 0.5
   masked_out_LM = torch.arange(L).unsqueeze(1) < torch.arange(L).unsqueeze(0)
   logits_BHLM += torch.where(masked_out_LM, -inf, 0)
   weights_BHLM = torch.softmax(logits_BHLM)
   wtd_values_BLHK = torch.einsum('BMHK,BHLM->BLHK', value_BMHK, logits_BHLM)
   out_BLD = torch.einsum('BLHK,HKD->BLD', wtd_values_BLHK, params.w_o_HKD)
   return out_BLD

