import jax
import jax.numpy as jnp
import einops 

B = 2 # batch_sz
L = 1 # seq_len: 1 during decoding, equal to M. so L <= M
M = 7 # mem_len: sequence length being attended to 
D = 12 # model_dim: also known as d_model/embedding_dim. vocab size -> model_dim
V = 9 # vocab_size
F = 48 # ffn dim
H = 3 # attention heads
K = 4 # D/H, size of each attention key or value

def attn(x_BMD: jax.Array, w_qkv_3DD: jax.Array, w_out_DD: jax.Array, training: bool = False):
  
    x_BLD = x_BMD
    if not training:
        x_BLD = x_BMD[:, -1:, :] # take just the last token


    # split W_qkv
    w_qkv_3DHK = einops.rearrange(w_qkv_3DD, 'THREE D (H K) -> THREE D H K', H=H, K=K)
    w_q_DHK, w_k_DHK, w_v_DHK = w_qkv_3DHK[0], w_qkv_3DHK[1], w_qkv_3DHK[2]

    # split into heads

    # project into query/key/value
    query_BLHK = jnp.einsum('BLD,DHK->BLHK', x_BLD, w_q_DHK)
    key_BMHK = jnp.einsum('BMD,DHK->BMHK', x_BMD, w_k_DHK)
    value_BMHK = jnp.einsum('BMD,DHK->BMHK', x_BMD, w_v_DHK)
   
    # compute cos similarity over B and H. result matrix should be LM (would be MM for training)

    similarity_score_BHLM = jnp.einsum('BLHK,BMHK->BHLM', query_BLHK, key_BMHK)
    
    b, l, h, k = query_BLHK.shape
   
    scaled_score_BHLM = similarity_score_BHLM / (k**0.5) # scale by attention k/v dim

    # causal mask ? i forgot

    block_upper_LM = jnp.triu(jnp.ones((L, M)), k=1) # triu takes in a matrix as input
    causal_mask_LM = jnp.where(block_upper_LM == 1, -jnp.inf, 0.0) # -inf for blocked values, 0 otherwise

    causal_mask_BHLM = einops.rearrange(causal_mask_LM, 'L M -> 1 1 L M')

    
    masked_score_BHLM = scaled_score_BHLM + causal_mask_BHLM 

    softmaxed_score_BHLM = jax.nn.softmax(masked_score_BHLM, axis=-1)


    weights_BLHK = jnp.einsum('BHLM,BMHK->BLHK', softmaxed_score_BHLM, value_BMHK) # dot over BH to BHLK, reshape to BL
    
    weights_BLD = einops.rearrange(weights_BLHK, 'B L H K -> B L (H K)')
    
    attn_out_BLD = jnp.einsum('BLD,DD->BLD', weights_BLD, w_out_DD)

    return attn_out_BLD



if __name__ == "__main__":

    key = jax.random.PRNGKey(42)

    
    # Check available devices
    devices = jax.devices()
    print(f"Available devices: {devices}")
    
    # Use GPU if available, otherwise CPU
    device = jax.devices('gpu')[0] if jax.devices('gpu') else jax.devices('cpu')[0]
    print(f"Using device: {device}")
    
    # Create matrices on the specified device
    with jax.default_device(device):
        x_BMD = jax.random.normal(key, shape=(B,M,D))
        w_qkv_3DD = jax.random.normal(key, shape=(3,D,D))
        w_out_DD = jax.random.normal(key, shape=(D,D))

        result = attn(x_BMD, w_qkv_3DD, w_out_DD)


    print(f"Result: {result}")

