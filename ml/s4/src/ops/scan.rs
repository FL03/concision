/*
    Appellation: scan <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::params::SSMStore;
use ndarray::prelude::{Array1, Array2, ArrayView1};
use ndarray_linalg::error::LinalgError;
use ndarray_linalg::{vstack, Scalar};
use num::Float;

///
// TODO: Allow the scan's state to be returned for caching in the S4 model
pub fn scan_ssm<T>(
    a: &Array2<T>,
    b: &Array2<T>,
    c: &Array2<T>,
    u: &Array2<T>,
    x0: &Array1<T>,
) -> Result<Array2<T>, LinalgError>
where
    T: Scalar,
{
    let step = |xs: &mut Array1<T>, us: ArrayView1<T>| {
        *xs = a.dot(xs) + b.dot(&us);
        let y1 = c.dot(&xs.clone());
        Some(y1)
    };
    let scan: Vec<Array1<T>> = u.outer_iter().scan(x0.clone(), step).collect();
    vstack(scan.as_slice())
}

pub struct Scanner<'a, T = f64>
where
    T: Float,
{
    model: &'a mut SSMStore<T>,
}

impl<'a, T> Scanner<'a, T>
where
    T: Float,
{
    pub fn new(model: &'a mut SSMStore<T>) -> Self {
        Self { model }
    }

    pub fn model(&self) -> &SSMStore<T> {
        self.model
    }

    pub fn model_mut(&mut self) -> &mut SSMStore<T> {
        self.model
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::SSMStore;
    use ndarray::prelude::*;

    const FEATURES: usize = 3;

    #[test]
    fn test_scan() {
        let exp = array![[0.0], [5.0], [70.0]];

        let u = Array::range(0.0, FEATURES as f64, 1.0).insert_axis(Axis(1));
        let x0 = Array1::zeros(FEATURES); // Array1::<Complex<f64>>::zeros(FEATURES)

        let a = Array::range(0.0, (FEATURES * FEATURES) as f64, 1.0)
            .into_shape((FEATURES, FEATURES))
            .unwrap();
        let b = Array::range(0.0, FEATURES as f64, 1.0).insert_axis(Axis(1));
        let c = Array::range(0.0, FEATURES as f64, 1.0).insert_axis(Axis(0));

        let scan = scan_ssm(&a, &b, &c, &u, &x0).expect("");

        assert_eq!(&scan, &exp);

        let ssm = SSMStore::new(a, b, c, Array2::zeros((1, 1)));
        assert_eq!(&scan, &ssm.scan(&u, &x0).unwrap())
    }
}
