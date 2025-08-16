import jax
import jax.numpy as jnp

def matmul_inference():
    # Check available devices
    devices = jax.devices()
    print(f"Available devices: {devices}")
    
    # Use GPU if available, otherwise CPU
    device = jax.devices('gpu')[0] if jax.devices('gpu') else jax.devices('cpu')[0]
    print(f"Using device: {device}")
    
    # Create matrices on the specified device
    with jax.default_device(device):
        A = jnp.ones((123, 345))
        B = jnp.ones((345, 678))
        C = jnp.matmul(A, B)
        return C.sum()

if __name__ == "__main__":
    result = matmul_inference()
    print(f"Result: {result}")

