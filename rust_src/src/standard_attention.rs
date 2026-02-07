//! Standard O(N²) causal attention — control group for benchmarking.
//!
//! This is the textbook implementation that materializes the full
//! attention matrix in memory. It exists solely to provide a real
//! runtime baseline for comparing against the Waller operator.

/// Standard causal self-attention (O(N²) memory).
///
/// Allocates a full `[seq_len, seq_len]` scores matrix, applies
/// causal masking, row-wise softmax, then multiplies by V.
///
/// Arguments match `waller_operator` for direct comparison.
pub fn standard_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    head_dim: usize,
    scale: f32,
) -> Vec<f32> {
    // 1. Compute scaled dot-product scores: S[i][j] = scale * dot(Q[i], K[j])
    let mut scores = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len {
            if j <= i {
                // Causal: only attend to positions <= current
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[i * head_dim + d] * k[j * head_dim + d];
                }
                scores[i * seq_len + j] = dot * scale;
            } else {
                scores[i * seq_len + j] = f32::NEG_INFINITY;
            }
        }
    }

    // 2. Row-wise softmax
    for i in 0..seq_len {
        let row = &mut scores[i * seq_len..(i + 1) * seq_len];

        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for val in row.iter_mut() {
            *val = (*val - max).exp();
            sum += *val;
        }
        for val in row.iter_mut() {
            *val /= sum;
        }
    }

    // 3. Weighted sum: output[i] = sum_j( scores[i][j] * V[j] )
    let mut output = vec![0.0f32; seq_len * head_dim];
    for i in 0..seq_len {
        for j in 0..=i {
            let w = scores[i * seq_len + j];
            for d in 0..head_dim {
                output[i * head_dim + d] += w * v[j * head_dim + d];
            }
        }
    }

    output
}
