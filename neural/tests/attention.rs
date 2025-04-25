extern crate concision_core as cnc;
extern crate concision_neural as neural;

#[test]
#[ignore = "Not setup yet"]
fn test_attention() {
    use ndarray::prelude::*;
    use neural::layers::attention::ScaledDotProductAttention;

    let shape = (10, 10);
    let attention = ScaledDotProductAttention::new(shape, 0.1, 1.0);

    let query = Array2::<f32>::zeros((1, 10));
    let key = Array2::<f32>::zeros((1, 10));
    let value = Array2::<f32>::zeros((1, 10));
    // compute the attention scores
    let z_score = attention.attention(&query, &key, &value);

    assert_eq!(z_score.shape(), &[1, 10]);
}
