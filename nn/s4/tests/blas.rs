#[cfg(test)]
use ndarray::prelude::{array, Array1, Array2};
use ndarray_linalg::eig::Eig;
use num::complex::{Complex, ComplexFloat};
use num::Float;

fn round_to<T: Float>(a: T, precision: usize) -> T {
    let factor = T::from(10).unwrap().powi(precision as i32);
    (a * factor).round() / factor
}

#[test]
fn test_eig() {
    let a = array![[1.0, 2.0], [2.0, 1.0]];
    let (eig, eigval): (Array1<Complex<f64>>, Array2<Complex<f64>>) = a.eig().unwrap();

    let eig = eig.mapv(|i| Complex::new(i.re().round(), i.im().round()));
    let eigval = eigval.mapv(|i| Complex::new(round_to(i.re(), 8), round_to(i.im(), 8)));

    let exp_eig: Array1<Complex<f64>> = array![3.0, -1.0].mapv(|i| Complex::new(i, 0.0));
    let exp_eigval: Array2<Complex<f64>> =
        array![[0.70710678, -0.70710678], [0.70710678, 0.70710678]].mapv(|i| Complex::new(i, 0.0));

    assert_eq!(eig, exp_eig);
    assert_eq!(eigval, exp_eigval);
}
