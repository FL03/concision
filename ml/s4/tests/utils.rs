#[cfg(test)]
extern crate concision_s4;
use concision_s4 as s4;

use ndarray::prelude::*;
use s4::prelude::{scanner, SSMStore};

#[test]
fn test_scan() {
    let features = 2;

    let u = Array2::ones((10, features));
    let x0 = Array1::ones(features);

    let ssm = SSMStore::<f64>::from_features(features);
    let (a, b, c, _d) = ssm.clone().into();
    let scan1 = scanner(&a, &b, &c, &u, &x0);

    let scan2 = ssm.scan(&u, &x0).unwrap();

    assert_eq!(scan1, scan2);
}
