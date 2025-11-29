/*
    Appellation: params <test>
    Contrib: @FL03
*/
use concision_params::Params;

use approx::assert_abs_diff_eq;
use ndarray::prelude::*;

#[test]
fn test_params_forward() {
    let params = Params::<f64>::ones((3, 4));
    let input = array![1.0, 2.0, 3.0];
    // should be of shape 4: (d_in, d_out).t() * (d_in,) + (d_out,)
    let output = params.forward(&input);
    assert_eq!(output.dim(), 4);
    // output should be: $W.t() * x + b = [7.0, 7.0, 7.0, 7.0]$
    // where W = ones(3, 4) and b = ones(4)
    assert_abs_diff_eq!(output, array![7.0, 7.0, 7.0, 7.0], epsilon = 1e-3);
}
