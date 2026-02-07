//! Minimal GAE Usage Example
//!
//! Shows how to invoke the Waller Kernel for attention computation.

use gae::{WallerKernel, AttentionConfig};

fn main() {
// Configure attention
let config = AttentionConfig {
seq_len: 4096,
head_dim: 64,
num_heads: 12,
dtype: f16,
};



// Initialize kernel
let kernel = WallerKernel::new(&config).expect("Failed to init kernel");

// Prepare Q, K, V tensors (your data here)
let q = Tensor::randn(&[batch, seq_len, head_dim]);
let k = Tensor::randn(&[batch, seq_len, head_dim]);
let v = Tensor::randn(&[batch, seq_len, head_dim]);

// Run fused attention - 2 HBM trips instead of 12
let output = kernel.forward(&q, &k, &v);

// Output is bit-exact deterministic
println!("Attention output shape: {:?}", output.shape());
}
