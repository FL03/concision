/*
    Appellation: softmax <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::func::activate::Gradient;
use ndarray::prelude::{Array, Axis, Dimension, NdFloat};
use ndarray::RemoveAxis;
use num::Float;
use serde::{Deserialize, Serialize};

#[derive(
    Clone, Copy, Debug, Default, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize,
)]
pub struct Softmax {
    axis: Option<usize>,
}

impl Softmax {
    pub fn new(axis: Option<usize>) -> Self {
        Self { axis }
    }

    pub fn axis(&self) -> Option<usize> {
        self.axis
    }

    pub fn softmax<T, D>(args: Array<T, D>) -> Array<T, D>
    where
        D: Dimension,
        T: Float,
    {
        let denom = args.mapv(|x| x.exp()).sum();
        args.mapv(|x| x.exp() / denom)
    }

    pub fn softmax_axis<T, D>(&self, args: Array<T, D>) -> Array<T, D>
    where
        T: NdFloat,
        D: Dimension + RemoveAxis,
    {
        let exp = args.mapv(|x| x.exp());
        if let Some(axis) = self.axis {
            let denom = exp.sum_axis(Axis(axis));
            exp / denom
        } else {
            let denom = exp.sum();
            exp / denom
        }
    }
}

// impl<T, D> Activate<T, D> for Softmax
// where
//     D: Dimension + RemoveAxis,
//     T: NdFloat,
// {
//     fn activate(&self, x: &Array<T, D>) -> Array<T, D> {
//         let exp = x.mapv(|x| x.exp());
//         if let Some(axis) = self.axis {
//             let denom = exp.sum_axis(Axis(axis));
//             exp / denom
//         } else {
//             let denom = exp.sum();
//             exp / denom
//         }
//     }
// }

impl<T, D> Gradient<T, D> for Softmax
where
    D: Dimension + RemoveAxis,
    T: NdFloat,
{
    fn gradient(&self, args: &Array<T, D>) -> Array<T, D> {
        let exp = args.mapv(|x| x.exp());
        if let Some(axis) = self.axis {
            let denom = exp.sum_axis(Axis(axis));
            exp / denom
        } else {
            let denom = exp.sum();
            exp / denom
        }
    }
}

impl<T, D> Fn<(&Array<T, D>,)> for Softmax
where
    D: Dimension + RemoveAxis,
    T: NdFloat,
{
    extern "rust-call" fn call(&self, args: (&Array<T, D>,)) -> Self::Output {
        let exp = args.0.mapv(|x| x.exp());
        if let Some(axis) = self.axis {
            let denom = exp.sum_axis(Axis(axis));
            exp / denom
        } else {
            let denom = exp.sum();
            exp / denom
        }
    }
}

impl<T, D> FnMut<(&Array<T, D>,)> for Softmax
where
    D: Dimension + RemoveAxis,
    T: NdFloat,
{
    extern "rust-call" fn call_mut(&mut self, args: (&Array<T, D>,)) -> Self::Output {
        let exp = args.0.mapv(|x| x.exp());
        if let Some(axis) = self.axis {
            let denom = exp.sum_axis(Axis(axis));
            exp / denom
        } else {
            let denom = exp.sum();
            exp / denom
        }
    }
}

impl<T, D> FnOnce<(&Array<T, D>,)> for Softmax
where
    D: Dimension + RemoveAxis,
    T: NdFloat,
{
    type Output = Array<T, D>;

    extern "rust-call" fn call_once(self, args: (&Array<T, D>,)) -> Self::Output {
        let exp = args.0.mapv(|x| x.exp());
        if let Some(axis) = self.axis {
            let denom = exp.sum_axis(Axis(axis));
            exp / denom
        } else {
            let denom = exp.sum();
            exp / denom
        }
    }
}
