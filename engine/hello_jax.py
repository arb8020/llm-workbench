import jax.numpy as jnp

def matmul_inference():
    # Your JAX matmul code
    A = jnp.ones((1000, 1000))
    B = jnp.ones((1000, 1000))
    C = jnp.matmul(A, B)
    return C.sum()

if __name__ == "__main__":
    result = matmul_inference()
    print(f"Result: {result}")

