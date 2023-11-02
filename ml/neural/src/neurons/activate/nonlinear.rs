/*
    Appellation: nonlinear <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::Activator;
use ndarray::prelude::{Array, Array1};

pub fn softmax<T>(args: Array1<T>) -> Array1<T>
where
    T: num::Float,
{
    let denom = args.mapv(|x| x.exp()).sum();
    args.mapv(|x| x.exp() / denom)
}

pub struct ReLU;

impl ReLU {
    pub fn compute<T: PartialOrd + num::Zero>(x: T) -> T {
        if x > T::zero() {
            x
        } else {
            T::zero()
        }
    }
}

impl<T, D> Activator<Array<T, D>> for ReLU
where
    D: ndarray::Dimension,
    T: num::Float,
{
    fn rho(x: Array<T, D>) -> Array<T, D> {
        x.mapv(|x| Self::compute(x))
    }
}

pub struct Sigmoid;

impl Sigmoid {
    pub fn compute<T: num::Float>(x: T) -> T {
        T::one() / (T::one() + (-x).exp())
    }
}

impl<T, D> Activator<Array<T, D>> for Sigmoid
where
    D: ndarray::Dimension,
    T: num::Float,
{
    fn rho(x: Array<T, D>) -> Array<T, D> {
        x.mapv(|x| Self::compute(x))
    }
}
pub struct Softmax;

impl Softmax {
    pub fn new() -> Self {
        Self
    }
}

impl<T, D> Activator<Array<T, D>> for Softmax
where
    D: ndarray::Dimension,
    T: num::Float,
{
    fn rho(x: Array<T, D>) -> Array<T, D> {
        let denom = x.mapv(|x| x.exp()).sum();
        x.mapv(|x| x.exp() / denom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use computare::prelude::RoundTo;
    use ndarray::array;

    #[test]
    fn test_softmax() {
        let exp = array![0.09003057, 0.24472847, 0.66524096];
        let args = array![1.0, 2.0, 3.0];
        let res = Softmax::rho(args).mapv(|i| i.round_to(8));
        assert_eq!(res, exp);
    }
}
