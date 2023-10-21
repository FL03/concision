/*
    Appellation: nonlinear <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use ndarray::prelude::Array1;

pub fn softmax<T>(args: Array1<T>) -> Array1<T>
where
    T: num::Float,
{
    let denom = args.mapv(|x| x.exp()).sum();
    args.mapv(|x| x.exp() / denom)
}

pub struct Softmax {
    args: Array1<f64>,
}

impl Softmax {
    pub fn new(args: Array1<f64>) -> Self {
        Self { args }
    }

    pub(crate) fn denom(&self) -> f64 {
        self.args.mapv(|x| x.exp()).sum()
    }

    pub fn compute(&self) -> Array1<f64> {
        self.args.mapv(|x| x.exp() / self.denom())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use computare::prelude::RoundTo;

    #[test]
    fn test_softmax() {
        let args = Array1::from(vec![1.0, 2.0, 3.0]);
        let res = softmax(args).mapv(|i| i.round_to(8));
        assert_eq!(res, Array1::from(vec![0.09003057, 0.24472847, 0.66524096]));
    }
}
