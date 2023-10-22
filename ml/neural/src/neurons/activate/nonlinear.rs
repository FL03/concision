/*
    Appellation: nonlinear <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::Activator;
use ndarray::prelude::Array1;

pub fn softmax<T>(args: Array1<T>) -> Array1<T>
where
    T: num::Float,
{
    let denom = args.mapv(|x| x.exp()).sum();
    args.mapv(|x| x.exp() / denom)
}

pub struct Sigmoid;

impl Sigmoid {
    pub fn compute(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }
}

impl<T> Activator<T> for Sigmoid
where
    T: num::Float,
{
    fn rho(x: T) -> T {
        T::one() / (T::one() + (-x).exp())
    }
}

pub struct Softmax;

impl Softmax {
    pub fn new() -> Self {
        Self
    }
}

impl<T> Activator<Array1<T>> for Softmax
where
    T: num::Float,
{
    fn rho(x: Array1<T>) -> Array1<T> {
        let denom = x.mapv(|x| x.exp()).sum();
        x.mapv(|x| x.exp() / denom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use computare::prelude::RoundTo;

    #[test]
    fn test_softmax() {
        let exp = Array1::from(vec![0.09003057, 0.24472847, 0.66524096]);
        let args = Array1::from(vec![1.0, 2.0, 3.0]);
        let res = Softmax::rho(args).mapv(|i| i.round_to(8));
        assert_eq!(res, exp);
    }
}
