/*
    Appellation: Tensor <test>
    Contrib: @FL03
*/
extern crate concision_tensor as cnc;
use cnc::Tensor;

#[test]
fn test_ones_and_zeros() {
    // weights retain the given shape (d_in, d_out)
    // bias retains the shape (d_out,)
    let ones = Tensor::<f64>::ones((3, 4));
    assert_eq!(ones.dim(), (3, 4));
    // weights retain the given shape (d_in, d_out)
    // bias retains the shape (d_out,)
    let zeros = Tensor::<f64>::zeros((3, 4));
    assert_eq!(zeros.dim(), (3, 4));
}

#[test]
#[cfg(feature = "init")]
fn test_tensor_init() {
    use concision_init::Initialize;

    let lecun = Tensor::<f64>::lecun_normal((3, 4));
    assert_eq!(lecun.dim(), (3, 4));

    let glorot_norm = Tensor::<f64>::glorot_normal((3, 4));
    assert_eq!(glorot_norm.dim(), (3, 4));
    assert_ne!(lecun, glorot_norm);
    let glorot_uniform = Tensor::<f64>::glorot_uniform((3, 4)).expect("glorot_uniform failed");
    assert_eq!(glorot_uniform.dim(), (3, 4));
    assert_ne!(lecun, glorot_uniform);
    assert_ne!(glorot_norm, glorot_uniform);
    let truncnorm = Tensor::<f64>::truncnorm((3, 4), 0.0, 1.0).expect("truncnorm failed");
    assert_eq!(truncnorm.dim(), (3, 4));
}
