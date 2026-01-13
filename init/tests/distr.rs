/*
   Appellation: random <test>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use concision_init::RandTensor;
use concision_init::distr::LecunNormal;
use ndarray::prelude::*;

#[test]
fn test_init_ext() {
    let shape = [3, 3];
    let seed = 0u64;
    let a = Array2::<f64>::stdnorm(shape);
    let b = Array2::<f64>::stdnorm_from_seed(shape, seed);

    assert_eq!(a.shape(), shape);
    assert_eq!(a.shape(), b.shape());
}

#[test]
fn test_lecun_normal() {
    let shape = (3, 3);

    let distr = LecunNormal::new(shape.0);

    let bnd = 2f64 * distr.std_dev::<f64>();

    let arr = Array2::<f64>::lecun_normal(shape);

    assert!(arr.iter().all(|&x| x >= -bnd && x <= bnd));

    assert_eq!(arr.dim(), shape);
}

#[test]
fn test_truncnorm() {
    let (mean, std) = (0f64, 2f64);
    let bnd = 2f64 * std;
    let shape = (3, 3);
    let arr = Array::truncnorm(shape, mean, std).unwrap();
    assert!(arr.iter().all(|&x| x >= -bnd && x <= bnd));
}
