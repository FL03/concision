/*
    Appellation: params <test>
    Contrib: @FL03
*/
use concision_params::Params;

use ndarray::array;

/*
    Verify the dimensionality of the output of forward propagation
*/
#[test]
fn test_params_fwd_dimensionality() {
    let params = Params::<f64>::ones((3, 4));
    let input = array![1.0, 2.0, 3.0];
    // should be of shape 4:
    let output = params.forward(&input);
    assert_eq!(output.dim(), 4); // (in, out).t() * (in,) + (out,)
    assert_eq!(output, array![7.0, 7.0, 7.0, 7.0]); // w.t() * x + b = [7.0, 7.0, 7.0, 7.0]
}
