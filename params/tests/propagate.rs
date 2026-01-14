/*
    Appellation: params <test>
    Contrib: @FL03
*/
use concision_params::Params;

use ndarray::{Ix2, array};

/*
    Verify the dimensionality of the output of forward propagation.

    Given some parameters of shape (in, out) and given some input Params of shape (in, out) and
    an input of shape (..., in,), the output should be of shape (out,...).
*/
#[test]
fn test_params_fwd_dimensionality() {
    // define the input
    let input = array![1.0, 2.0, 3.0];
    // initialize the params
    let params = Params::<f64, Ix2>::ones((3, 4));
    // complete a forward pass
    let y = params.forward(&input);
    // verify the results
    assert_eq!(y.dim(), 4);
    assert_eq!(y, array![7.0, 7.0, 7.0, 7.0]);
}
