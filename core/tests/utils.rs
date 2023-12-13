#[cfg(test)]
extern crate concision_core;

use concision_core::prelude::{linarr, now};
use ndarray::prelude::{array, Array2};

#[test]
fn test_linarr() {
    let args: Array2<f64> = linarr((2, 3)).unwrap();
    assert_eq!(&args, &array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
}

#[test]
fn test_timestamp() {
    let period = std::time::Duration::from_secs(1);
    let ts = now();
    assert!(ts > 0);
    std::thread::sleep(period);
    assert!(now() > ts);
}
