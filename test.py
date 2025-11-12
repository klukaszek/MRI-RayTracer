import os
import jax
import jax.numpy as jnp
import numpy as np
import psutil

print("=" * 60)
print("SYSTEM INFO")
print("=" * 60)
print(f"Physical RAM: {psutil.virtual_memory().total / 1e9:.2f} GB")
print(f"Available RAM: {psutil.virtual_memory().available / 1e9:.2f} GB")
print(f"CPU cores: {psutil.cpu_count()}")

print("\n" + "=" * 60)
print("JAX INFO")
print("=" * 60)
print(f"JAX version: {jax.__version__}")
print(f"Backend: {jax.default_backend()}")
print(f"Devices: {jax.devices()}")
print(f"Local devices: {jax.local_device_count()}")

print("\n" + "=" * 60)
print("ENV VARS")
print("=" * 60)
for key in ['XLA_PYTHON_CLIENT_PREALLOCATE', 'XLA_PYTHON_CLIENT_MEM_FRACTION', 
            'JAX_PLATFORMS', 'CUDA_VISIBLE_DEVICES', 'JAX_ENABLE_X64']:
    print(f"{key}: {os.environ.get(key, 'NOT SET')}")

print("\n" + "=" * 60)
print("MEMORY TEST")
print("=" * 60)
try:
    # Try to allocate same size as your data
    test_size = (200, 4, 240, 240, 155)
    print(f"Attempting to allocate {np.prod(test_size) * 4 / 1e9:.2f} GB...")
    
    arr = jnp.zeros(test_size, dtype=jnp.float32)
    print("✓ JAX allocation succeeded")
    del arr
except Exception as e:
    print(f"✗ JAX allocation failed: {e}")
