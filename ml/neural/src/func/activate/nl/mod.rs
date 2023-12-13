/*
    Appellation: nonlinear <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # NonLinear Activation Functions
//!
//!
pub use self::{relu::*, sigmoid::*, softmax::*, tanh::*, utils::*};

pub(crate) mod relu;
pub(crate) mod sigmoid;
pub(crate) mod softmax;
pub(crate) mod tanh;

pub(crate) mod utils {
    use ndarray::prelude::{Array, Axis, Dimension, NdFloat};
    use ndarray::RemoveAxis;
    use num::{Float, Zero};

    pub fn relu<T>(args: &T) -> T
    where
        T: Clone + PartialOrd + Zero,
    {
        if args > &T::zero() {
            args.clone()
        } else {
            T::zero()
        }
    }

    pub fn sigmoid<T>(args: &T) -> T
    where
        T: Float,
    {
        T::one() / (T::one() + (-args.clone()).exp())
    }

    pub fn softmax<T, D>(args: &Array<T, D>) -> Array<T, D>
    where
        D: Dimension,
        T: Float,
    {
        let denom = args.mapv(|x| x.exp()).sum();
        args.mapv(|x| x.exp() / denom)
    }

    pub fn softmax_axis<T, D>(args: &Array<T, D>, axis: Option<usize>) -> Array<T, D>
    where
        D: Dimension + RemoveAxis,
        T: NdFloat,
    {
        let exp = args.mapv(|x| x.exp());
        if let Some(axis) = axis {
            let denom = exp.sum_axis(Axis(axis));
            exp / denom
        } else {
            let denom = exp.sum();
            exp / denom
        }
    }

    pub fn tanh<T>(args: &T) -> T
    where
        T: Float,
    {
        args.tanh()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::Activate;
    use computare::prelude::RoundTo;
    use ndarray::array;

    #[test]
    fn test_relu() {
        let exp = array![0.0, 0.0, 3.0];
        let args = array![-1.0, 0.0, 3.0];

        let res = ReLU::new().activate(&args);
        assert_eq!(res, exp);
        assert_eq!(ReLU(&args), exp);
    }

    #[test]
    fn test_sigmoid() {
        let exp = array![0.73105858, 0.88079708, 0.95257413];
        let args = array![1.0, 2.0, 3.0];

        let res = Sigmoid::new().activate(&args).mapv(|i| i.round_to(8));
        assert_eq!(res, exp);
        let res = Sigmoid(&args).mapv(|i| i.round_to(8));
        assert_eq!(res, exp);
    }

    #[test]
    fn test_softmax() {
        let exp = array![0.09003057, 0.24472847, 0.66524096];
        let args = array![1.0, 2.0, 3.0];

        let res = Softmax::new(None).activate(&args).mapv(|i| i.round_to(8));
        assert_eq!(res, exp);
        let res = Softmax::new(None)(&args).mapv(|i| i.round_to(8));
        assert_eq!(res, exp);
    }

    #[test]
    fn test_tanh() {
        let exp = array![0.76159416, 0.96402758, 0.99505475];
        let args = array![1.0, 2.0, 3.0];

        let res = Tanh::new().activate(&args).mapv(|i| i.round_to(8));
        assert_eq!(res, exp);
        let res = Tanh(&args).mapv(|i| i.round_to(8));
        assert_eq!(res, exp);
    }
}
