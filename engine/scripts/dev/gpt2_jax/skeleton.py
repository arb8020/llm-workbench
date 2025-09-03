#!/usr/bin/env python3
"""
Minimal GPT-2 JAX implementation skeleton.

Students should implement gpt2_forward() to match HuggingFace GPT-2 logits.

Usage:
    python engine/scripts/dev/hello_gpt2_jax_skeleton.py
"""

import jax
import jax.numpy as jnp
import einops
from typing import Dict, Optional, Callable
from dataclasses import dataclass
from jax import Array
from engine.core.utils.weights import load_gpt2_weights, download_gpt2_weights, load_and_print_gpt2_weights_jax, validate_gpt2_weights

"""
B: batch size
L: sequence length
M: memory length (length of sequence being attended to)
D: model dimension (sometimes called d_model or embedding_dim)
V: vocabulary size
F: feed-forward subnetwork hidden size
H: number of attention heads in a layer
K: size of each attention key or value (sometimes called d_kv)
"""

@dataclass(frozen=True)
class GPT2Config:
    """Configuration for GPT-2 model."""
    vocab_size: int = 50257
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_positions: int = 1024
    layer_norm_epsilon: float = 1e-5
    use_cache: bool = True
    training: bool = False
    freqs_cis: Optional[Array] = None  
    
    def __post_init__(self):
        """Validate configuration parameters."""
        assert self.d_model % self.n_heads == 0, f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        assert self.vocab_size > 0, "vocab_size must be positive"
        assert self.n_layers > 0, "n_layers must be positive"


def _validate_ffn_shapes(x_BLD: jnp.ndarray, 
                        weight_in_DF: jnp.ndarray, 
                        bias_in_F: jnp.ndarray,
                        weight_out_FD: jnp.ndarray, 
                        bias_out_D: jnp.ndarray):
    
    D = x_BLD.shape[-1]  
    assert weight_in_DF.shape[0] == D, "weight_in_DF's first dimension must match x_BLD's last dimension"
    F = weight_in_DF.shape[1]  
    assert bias_in_F.shape[0] == F, "bias_in_F dimension must match weight_in_DF's second dimension"
    
    assert weight_out_FD.shape[0] == F, "weight_out_FD's first dimension must match weight_in_DF's second dimension"
    assert weight_out_FD.shape[1] == D, "weight_out_FD's second dimension must match x_BLD's last dimension"
    assert bias_out_D.shape[0] == D, "bias_out_D dimension must match x_BLD's last dimension"

def _validate_linear_shapes(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> None:
    assert x.shape[-1] == weight.shape[0], f"x shape {x.shape} incompatible with weight shape {weight.shape}"
    assert weight.shape[1] == bias.shape[0], f"weight shape {weight.shape} incompatible with bias shape {bias.shape}"

def _validate_layer_norm_shapes(x_BLD: jnp.ndarray, gamma: jnp.ndarray, beta: jnp.ndarray):
    assert gamma.shape[-1] == x_BLD.shape[-1], "gamma's last dimension must match x_BLD's last dimension"
    assert beta.shape[-1] == x_BLD.shape[-1], "beta's last dimension must match x_BLD's last dimension"

def _validate_attention_shapes(x_BMD: jnp.ndarray, w_qkv_3DD: jnp.ndarray, b_qkv_3D: jnp.ndarray, 
                             w_out_DD: jnp.ndarray, b_out_D: jnp.ndarray, config: GPT2Config):
    """Validate all attention layer input shapes match expected dimensions."""
    B, M, D = x_BMD.shape
    
    # Validate input
    assert len(x_BMD.shape) == 3, f"x_BMD must be 3D, got shape {x_BMD.shape}"
    assert D == config.d_model, f"x_BMD last dim {D} must match config.d_model {config.d_model}"
    
    # Validate Q,K,V weights
    assert w_qkv_3DD.shape == (3, D, D), f"w_qkv_3DD shape {w_qkv_3DD.shape} must be (3, {D}, {D})"
    assert b_qkv_3D.shape == (3, D), f"b_qkv_3D shape {b_qkv_3D.shape} must be (3, {D})"
    
    # Validate output projection
    assert w_out_DD.shape == (D, D), f"w_out_DD shape {w_out_DD.shape} must be ({D}, {D})"
    assert b_out_D.shape == (D,), f"b_out_D shape {b_out_D.shape} must be ({D},)"
    
    # Validate head configuration
    assert D % config.n_heads == 0, f"d_model {D} must be divisible by n_heads {config.n_heads}"
    head_dim = D // config.n_heads
    assert head_dim > 0, f"Head dimension {head_dim} must be positive"
    
    print(f"✅ Attention shapes validated: input={x_BMD.shape}, heads={config.n_heads}, head_dim={head_dim}")


"""
layers:

ok so we have the input obviously of some amount of batches, B
im not sure if the next dimension is sequence or memory length, probably L
after that, the input has to be of dimension V, since its just been tokenized
so input_BLV -> embedded_input_BLD is the first transformation
this would be done by wte linear layer, i think with no bias
after wte, we positionally embed, so we would use wpe to go from 
embedded_BLD -> positional_BLD

now, positional_BLD goes through 12 blocks of transformer
let's call each blocks' input 
block_input_BLD
we want to have a residual stream, so we save block_input_BLD
the residual stream means that we can get more stable training, because learning to do 'add 0' is easier than learning the identity function
each layer only has to 'contribute' some new information, rather than spending compute reconstructing the input
its kind of like playing telephone. having to reconstruct the initial signal every time might get lost
but if we could write down what the first person said, pass it along as a slip of paper, and everyone else could just add their sticky note to the paper, it would be easier to communicate

the two layers we really care about in the block are the attention layer and the MLP layer
token sequences are, obviously, sequential. so we need something that can process multiple vectors of input
TODO: CONNECT BETTER
if we put this straight into the MLP processing layer, it essentially becomes a token lookup table
to illustrate this, notice that the only thing that can be learned about a given token is what the next token to output is
what's the cleanest way to get the most recent token to have information about what came before it? 
when we process a sentence 'my dog ate the ...', we choose the next word to say based on all previous words
so we have to get the syntactical and semantic contextual information compressed into the word 'the'
TODO: CONNECT BETTER
one way you might think to do this is to compress the information of 'my' into 'dog'
then, 'dog', with the compressed info of 'my' can get to 'ate'
notice that this is a compounding/multiplicative effect
which is subject to information loss, and more worryingly explosion/vanishing of the information from a new datapoint
since the gradient update of the new datapoint now has to flow through all of these tokens
a very small gradient might disappear into 0, as small numbers compound to even smaller 
what we really do, as humans, is more of like a lookup across all previous tokens in our working memory, at each position
this sidesteps that explosion/vanishing problem
when we think about the next word to say, we're taking into account the fact that we're referring to 'my' dog, we're thinking about tokens related to dogs, related to eating, and therefore related to 'dogs eating', and we know that its an item described as singular due to 'the'
so we need a mechanism that can do this kind of comparison 
for a given token, we want to know how important the previous tokens are, across some amount of useful axes
'the' might be a useful token in a syntactical way, since it restricts the space of tokens we might want to generate into singular nouns
'dog' and 'ate' might also be useful in a semantic meaning dimension, since they constrict the space of what kind of singular noun to generate, one that is something a dog may eat
if we had even more information in the sentence, 'my' could have been useful
were it to have been mentioned earlier in the sequence 'i bought a treat', the token 'treat' might be much more important, since now a relation to 'i/my' has been established
in any case, we first get these positional attention scores for how related a given word is to another
we can model this as sort of a fuzzy hashmap lookup, how 'similar/related' is 'the' to all of the previous tokens, across those dimensions of syntax/semantics/etc - where 'the' is our query, and the keys are the previous tokens
two vectors can be called 'similar' based on the angle between them
if we take the dot product of two vectors, and divide by their magnitude, we get the cosine of the angle between them
we usually decide to not divide by the magnitude, because the magnitude of the vector is useful information we want the model to be able to learn along
instead, we correct the scale of the dot product by dividing by the square root of the dimension of the key/query vectors
this is because as dimensionality increases, the dot products between the vectors might get really big, because the variance of each vector's magnitude in a given dimension is high
since these are raw values and don't correspond to 'what % of attention should this value get', we use softmax to turn the vector of raw values into one that has all non negative values, and they add up to 1, a probability distribution

NOTE ON SOFTMAX. SKIPPABLE
softmax itself comes from some interesting information theory, literally the element of surprise. 
we can think of the problem as taking our raw vector from before, x, with elements x_i, and turning it into a new vector p, where sum(p_i) = 1 and no p_i < 0. 
since we want to maximally conserve information, over the space of vectors p that fulfill the above conditions, we want to choose the vector with the highest entropy
the most 'random' or 'noisy' vector is the one that is least likely to have any other biases that have come with it, that are not x
so we also want the new vector p to be the one in the space of all such possible vectors P to have max(entropy(p)).
we can define entropy rigorously as the expected 'surprise'. 
this is pretty intuitive. suppose i flip a coin and i get heads. i shouldn't be 'surprised' because its one of two outcomes. 
but if i have 100 blue balls and 2 red balls, and i draw a red one, i should be 'more surprised'
we might want to reach for the inverse of the probability of the event, 1/p, but this kind of breaks at 0 so we throw a log transform at it and call it a day
since we want the expected surprise, we take the surprise of each event, and multiply it by the probability of the event
so we have sum(p_i * log(1/p_i))
log(1/x) = -log(x)
so sum(p_i * -log(p_i))
and we have -(sum(p_i) * log(p_i)) is the comparator between two different vectors p that might both be probability distributions of f
if we don't somehow account for the original vector f, we end up with a uniform distribution since that would have maximum entropy
suppose f = [1,2,3], we know that we need our p vector to follow this trend where the first element is smaller than the second which is smaller than the third
we can think of f_i as representing the energy of a given state. and p_i is the probablity of being in some state i
so we can ground our entropy maximization function in the constraint that sum(p_i * f_i) must stay equivalent to some constant
think of it like this: we want to find an arrangement of a rectangle that has a fixed perimeter, but maximizes area
we don't necessarily care what the maximum area is, but we can still use this constraint to arrive at a formalization/function of the area
let a, b be the sides of the rectangle, and let A = ab, and perimeter P = 2a+2b
so we want to maximize A with respect to the constraint that 2a + 2b = P
from this constraint, we find that b = P/2-a
so we can rewrite A as a function of just a, where A = (P/2-a)a
or A = aP/2 - a^2
taking the partial derivative with respect to a, we find
dA/da = P/2-2a
0 = P/2-2a
2a = P/2
a = P/4
if a = P/4, and b = P/2-a, then b = P/4 as well
so even though we didn't care what the perimeter P was, it became a useful constraint in order to solve our problem with respect to the terms we have
in the same way, solving our optimization problem
maximize entropy -sum(p_i * log(p_i))
p_i >= 0
sum(p_i) = 1
sum(p_i * f_i)) = beta
i won't do the math here but we get 
p_i = exp(beta * f_i)/sum(exp(beta * f_j)) 
so we might have 
index 0 of the p vector of the original f [1, 2, 3] is
e^1/(e^1+e^2+e^3)
assuming beta = 1
note that beta can be expressed as 1/T, where T akin to a 'temperature' parameter
the idea of 'temperature' here basically implying that higher temperature leads to higher entropy in the derived version of the softmax here, since a lower beta constant entropy would get us closer to a uniform p vector distribution

END SOFTMAX TANGENT

so now, once we get how related they are, from this probability distribution, those tokens return some 'value' that represents, in the attention space, what it means, and we weight that value by our relation score
TODO: what is a value? you didn't explain it
so our new contextualized representation of 'the' is what's outputted by this attention layer

THIS MAY ALSO BE A TANGENT
note that during training, we actually process all of the tokens in the sequence with each other at once, rather than one at a time like when we want to generate the next token
so all we have to do during training is automatically force the query/key similarity values to be super negative for any tokens that wouldn't have existed
the first token can only pay attention to itself
then the second token pays attention to itself and the previous
and so on
this is essentially a lower triangular matrix of 0s, with the upper triangle being the maximally negative float of your ML framework
we do it this way because this is the best way to tell the softmax to output a zero for that position in the vector
THIS MAY ALSO BE A TANGENT

TODO: this section on MLPs is a bit scattered
now, the MLP essentially just learns some useful way of processing that information, and adding to the residual stream
TODO: handwavy. wdym 'add to residual stream' how is it added? literal addition? transformation? 
MLPs are very powerful. 
a very small one can learn to recognize handwritten digits at very low resolutions (like 28x28), by literally flattening out the images pixels, assigning the brightness score of each pixel 0-1, and then outputting a guess distribution over the digits 0-9.
we can think of the problem as essentially learning some mapping from the input vector space, above the space of dimension 784, the flat representation of the pixels, into the space of 10, the vector space of digits
more simply, suppose we need to learn some function y = f(x) from a bunch of datapoints
we can tune a linear function y = Wx + b to get a line of best fit over the datapoints
imagine this in 2 dimensions, mapping from the x dimension to the y dimension. 
we will never have something other than a straight line
so if our f(x) is x^2, our learning is capped to some error bound that we can never surpass
in order to learn it more closely, we need to bend the line somehow
we do this by applying a nonlinear function over the linear transformation, something like
y = nonlinear_fn(Wx+b), which can warp and bend the initial line into something that can hug x^2 better
a classic nonlinear function is y = max(0,x), as it introduces a 'kink' at 0 where it goes from a slope of 1 in the positive domain to a slope of 0 in the negative domainnow, if we stack this, we go from
y = nonlinear_fn(Wx+b)
to 
y = nonlinear_fn(W_1(W_0x + b_0) + b_1)
so now we get to have two bends in our line! 
as we stack up nonlinear layers, we basically get to warp space in more ways
bringing this back to the more abstract vector spaces, what we usually do is have a layer that goes from the input vector space dimension into a higher dimension, and then back down into our output dimension
think of the hidden dimension of the model as representing some amount of features or information in each dimension
the model likely has to learn more things than the amount of dimensions it has, so each dimension has multiple meanings
the MLP layer allows the model to have more 'room' to express features, and then come back down to the layer of the residual
we might think of it as the initial piece of paper everyone is passing through the residual stream as being in a caveman or primitive english
and then the person adding information can think in more sophisticated english/images/visualize, and then translate what they were able to learn in that language system back into the caveman english, to add their sticky note
this is also why we do the initial projection from vocabulary space into a model's embedding dimension. having to cram everything we know into only as many dimensions as we have tokens is very limiting
finally after a bunch of these layers, 12 in specific, we take the final residual stream's information, project it back down into the vocabulary space, and then model has a probability distribution over each of the tokens it could generate next, which is what we wanted!
TODO: rushed, forgot the final softmax step 

"""



def gelu(x):
    """Hendrycks & Gimpel (2016) https://arxiv.org/abs/1606.08415"""
    # Using the approximation formula for GELU
    # 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    cdf = 0.5 * (1.0 + jnp.tanh(
        jnp.sqrt(2.0 / jnp.pi) * (x + 0.044715 * jnp.power(x, 3))
    ))
    return x * cdf

def gelu_exact(x):
    """Hendrycks & Gimpel (2016) https://arxiv.org/abs/1606.08415"""

    return 0.5 * x * (1 + jnp.erf(x / jnp.sqrt(2.0)))

def project_and_embed(input_ids: jnp.ndarray, weights: Dict[str, Array], config: GPT2Config) -> jnp.ndarray:
    """Radford et al. (2019) https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf"""

    projected_BLD = weights['wte.weight'][input_ids]
    _, seq_len = input_ids.shape
    position_embeddings = weights['wpe.weight'][:seq_len]
    projected_embedded_BLD = projected_BLD + position_embeddings
    
    return projected_embedded_BLD

def layer_norm(x_BLD: jnp.ndarray, gamma: jnp.ndarray, beta: jnp.ndarray, epsilon: float = 1e-5) -> jnp.ndarray:
    """Ba et al. (2016) https://arxiv.org/abs/1607.06450"""

    _validate_layer_norm_shapes(x_BLD, gamma, beta)

    mean_BL1 = jnp.mean(x_BLD, axis=-1, keepdims=True)
    variance_BL1 = jnp.var(x_BLD, axis=-1, keepdims=True)

    demeaned_BLD = (x_BLD - mean_BL1) # BLD auto broadcasts over BL1
    demeaned_centered_BLD = demeaned_BLD / jnp.sqrt(variance_BL1 + epsilon)

    gamma_scaled_BLD = demeaned_centered_BLD * gamma
    beta_shifted_BLD = gamma_scaled_BLD + beta

    final_BLD = beta_shifted_BLD 

    return final_BLD


def ffn(x_BLD: jnp.ndarray, 
        weight_in_DF: jnp.ndarray, 
        bias_in_F: jnp.ndarray,
        weight_out_FD: jnp.ndarray, 
        bias_out_D: jnp.ndarray,
        activation_fn: Callable[[jnp.ndarray], jnp.ndarray]) -> jnp.ndarray:
    """Vaswani et al. (2017) https://arxiv.org/abs/1706.03762"""

    _validate_ffn_shapes(x_BLD, weight_in_DF, bias_in_F, weight_out_FD, bias_out_D) 
    hidden_BLF = linear(x_BLD, weight_in_DF, bias_in_F)
    activated_BLF = activation_fn(hidden_BLF)
    output_BLD = linear(activated_BLF, weight_out_FD, bias_out_D)

    return output_BLD


def linear(x: jnp.ndarray, weight: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    """Goodfellow et al. (2016) http://www.deeplearningbook.org"""
    _validate_linear_shapes(x, weight, bias)
    return x @ weight + bias

def multihead_attn(x_BMD: jax.Array, w_qkv_3DD: jax.Array, b_qkv_3D: jax.Array, w_out_DD: jax.Array, b_out_D: jax.Array, config: GPT2Config, training: bool = False) -> jnp.ndarray:
    """Vaswani et al. (2017) https://arxiv.org/abs/1706.03762"""
    
    # Validate shapes before proceeding
    _validate_attention_shapes(x_BMD, w_qkv_3DD, b_qkv_3D, w_out_DD, b_out_D, config)
    
    H = config.n_heads
    K = config.d_model // config.n_heads
    
    x_BLD = x_BMD
    if not training:
        x_BLD = x_BMD[:, -1:, :] # take just the last token

    # split W_qkv
    w_qkv_3DHK = einops.rearrange(w_qkv_3DD, 'THREE D (H K) -> THREE D H K', H=H, K=K)
    w_q_DHK, w_k_DHK, w_v_DHK = w_qkv_3DHK[0], w_qkv_3DHK[1], w_qkv_3DHK[2]

    # split biases
    b_q_DHK = einops.rearrange(b_qkv_3D[0], '(H K) -> H K', H=H, K=K)
    b_k_DHK = einops.rearrange(b_qkv_3D[1], '(H K) -> H K', H=H, K=K)
    b_v_DHK = einops.rearrange(b_qkv_3D[2], '(H K) -> H K', H=H, K=K)

    # project into query/key/value
    query_BLHK = jnp.einsum('BLD,DHK->BLHK', x_BLD, w_q_DHK) + b_q_DHK
    key_BMHK = jnp.einsum('BMD,DHK->BMHK', x_BMD, w_k_DHK) + b_k_DHK
    value_BMHK = jnp.einsum('BMD,DHK->BMHK', x_BMD, w_v_DHK) + b_v_DHK
   
    # compute cos similarity over B and H. result matrix should be LM (would be MM for training)
    similarity_score_BHLM = jnp.einsum('BLHK,BMHK->BHLM', query_BLHK, key_BMHK)
    
    b, l, h, k = query_BLHK.shape
   
    scaled_score_BHLM = similarity_score_BHLM / (k**0.5) # scale by attention k/v dim

    # causal mask
    l, m = scaled_score_BHLM.shape[-2:]

    block_upper_LM = jnp.triu(jnp.ones((l, m)), k=1) # triu takes in a matrix as input
    causal_mask_LM = jnp.where(block_upper_LM == 1, -jnp.inf, 0.0) # -inf for blocked values, 0 otherwise

    causal_mask_BHLM = einops.rearrange(causal_mask_LM, 'L M -> 1 1 L M')
    
    masked_score_BHLM = scaled_score_BHLM + causal_mask_BHLM 

    softmaxed_score_BHLM = jax.nn.softmax(masked_score_BHLM, axis=-1)

    weights_BLHK = jnp.einsum('BHLM,BMHK->BLHK', softmaxed_score_BHLM, value_BMHK) # dot over BH to BHLK, reshape to BL
    
    weights_BLD = einops.rearrange(weights_BLHK, 'B L H K -> B L (H K)')
    
    attn_out_BLD = jnp.einsum('BLD,DD->BLD', weights_BLD, w_out_DD) + b_out_D

    return attn_out_BLD

def gpt2_extract_block_weights(layer_idx: int, weights: Dict[str, Array]) -> Dict[str, Array]:
    """helper function to extract weights for a GPT2 block at given layer index"""
    return {
        'ln_1': {
            'weight': weights[f"h.{layer_idx}.ln_1.weight"],
            'bias': weights[f"h.{layer_idx}.ln_1.bias"]
        },
        'attn': {
            'c_attn': {
                'weight': weights[f"h.{layer_idx}.attn.c_attn.weight"],
                'bias': weights[f"h.{layer_idx}.attn.c_attn.bias"]
            },
            'c_proj': {
                'weight': weights[f"h.{layer_idx}.attn.c_proj.weight"],
                'bias': weights[f"h.{layer_idx}.attn.c_proj.bias"]
            }
        },
        'ln_2': {
            'weight': weights[f"h.{layer_idx}.ln_2.weight"],
            'bias': weights[f"h.{layer_idx}.ln_2.bias"]
        },
        'mlp': {
            'c_fc': {
                'weight': weights[f"h.{layer_idx}.mlp.c_fc.weight"],
                'bias': weights[f"h.{layer_idx}.mlp.c_fc.bias"]
            },
            'c_proj': {
                'weight': weights[f"h.{layer_idx}.mlp.c_proj.weight"],
                'bias': weights[f"h.{layer_idx}.mlp.c_proj.bias"]
            }
        }
    }

def gpt2_block(x_BLD: jnp.ndarray, layer_idx: int, weights: Dict[str, Array], config: GPT2Config) -> jnp.ndarray:
    """Radford et al. (2019) https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf"""
    
    block_weights = gpt2_extract_block_weights(layer_idx, weights)
    
    normed_x_BLD = layer_norm(x_BLD, block_weights['ln_1']['weight'], block_weights['ln_1']['bias'], config.layer_norm_epsilon)
    
    # GPT-2 c_attn contains concatenated Q,K,V weights - split them
    c_attn_weight = block_weights['attn']['c_attn']['weight']  # Shape: [D, 3*D]
    c_attn_bias = block_weights['attn']['c_attn']['bias']      # Shape: [3*D]
    
    # Split into Q, K, V
    d_model = config.d_model
    q_weight, k_weight, v_weight = jnp.split(c_attn_weight, 3, axis=1)
    q_bias, k_bias, v_bias = jnp.split(c_attn_bias, 3, axis=0)
    
    # Stack into w_qkv_3DD format: [3, D, D] and b_qkv_3D format: [3, D]
    w_qkv_3DD = jnp.stack([q_weight, k_weight, v_weight], axis=0)
    b_qkv_3D = jnp.stack([q_bias, k_bias, v_bias], axis=0)
    
    w_out_DD = block_weights['attn']['c_proj']['weight']
    b_out_D = block_weights['attn']['c_proj']['bias']
    
    attn_output_BLD = multihead_attn(normed_x_BLD, w_qkv_3DD, b_qkv_3D, w_out_DD, b_out_D, config)
    
    x_BLD = x_BLD + attn_output_BLD
    
    normed_x_BLD = layer_norm(x_BLD, block_weights['ln_2']['weight'], block_weights['ln_2']['bias'], config.layer_norm_epsilon)
    
    
    ffn_output_BLD = ffn(normed_x_BLD, 
                     block_weights['mlp']['c_fc']['weight'],
                     block_weights['mlp']['c_fc']['bias'],
                     block_weights['mlp']['c_proj']['weight'],
                     block_weights['mlp']['c_proj']['bias'], gelu)
    
    return x_BLD + ffn_output_BLD

def gpt2_forward(input_ids: jnp.ndarray, weights: Dict[str, Array], config: GPT2Config) -> jnp.ndarray:
    """Radford et al. (2019) https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf"""
    batch_size, seq_len = input_ids.shape
    vocab_size = config.vocab_size

    # Validate required weight keys exist
    required_keys = [
        'wte.weight', 'wpe.weight', 'ln_f.weight', 'ln_f.bias'
    ]
    # Add layer-specific keys
    for i in range(config.n_layers):
        layer_keys = [
            f'h.{i}.ln_1.weight', f'h.{i}.ln_1.bias',
            f'h.{i}.attn.c_attn.weight', f'h.{i}.attn.c_attn.bias',
            f'h.{i}.attn.c_proj.weight', f'h.{i}.attn.c_proj.bias',
            f'h.{i}.ln_2.weight', f'h.{i}.ln_2.bias',
            f'h.{i}.mlp.c_fc.weight', f'h.{i}.mlp.c_fc.bias',
            f'h.{i}.mlp.c_proj.weight', f'h.{i}.mlp.c_proj.bias'
        ]
        required_keys.extend(layer_keys)
    
    validate_gpt2_weights(weights, required_keys)

    projected_embedded_BLD = project_and_embed(input_ids, weights, config)
    x_BLD = projected_embedded_BLD

    for layer_idx in range(config.n_layers):
        x_BLD = gpt2_block(x_BLD, layer_idx, weights, config)

    x_BLD = layer_norm(x_BLD, 
                      weights['ln_f.weight'], 
                      weights['ln_f.bias'], 
                      config.layer_norm_epsilon)

    logits_BLV = jnp.einsum('BLD,VD->BLV', x_BLD, weights['wte.weight'])
    
    return logits_BLV


if __name__ == "__main__":
    
    # print real weights
    real_weights = load_and_print_gpt2_weights_jax() 
    
    config = GPT2Config()
    test_input = jnp.array([[15496, 995]])  # "Hello world" tokens
    
    logits = gpt2_forward(test_input, real_weights, config)
    
    
