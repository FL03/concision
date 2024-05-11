/*
    Appellation: mask <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use core::ops;
use ndarray::prelude::{Array, Array2, Dimension};
use ndarray::ScalarOperand;
use ndarray_rand::rand_distr::uniform::{SampleUniform, Uniform};
use ndarray_rand::RandomExt;
use num::traits::{Float, NumOps};
use smart_default::SmartDefault;
use strum::EnumIs;

#[derive(Clone, Debug, EnumIs, PartialEq, SmartDefault)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub enum Mask<T = f64> {
    Masked(Array2<T>),
    #[default]
    Unmasked,
}

impl<T> Mask<T>
where
    T: NumOps + ScalarOperand,
{
    pub fn forward(&self, data: &Array2<T>) -> Array2<T> {
        match self {
            Self::Masked(bias) => data + bias,
            Self::Unmasked => data.clone(),
        }
    }
}

impl<T> Mask<T>
where
    T: Float + SampleUniform,
{
    pub fn uniform(size: usize) -> Self {
        let ds = (T::from(size).unwrap()).sqrt();
        let dist = Uniform::new(-ds, ds);
        let mask = Array2::<T>::random((size, size), dist);
        Self::Masked(mask)
    }
}

impl<T> From<usize> for Mask<T>
where
    T: Float + SampleUniform,
{
    fn from(size: usize) -> Self {
        let ds = (T::from(size).unwrap()).sqrt();
        let dist = Uniform::new(-ds, ds);
        let mask = Array2::<T>::random((size, size), dist);
        Self::Masked(mask)
    }
}

impl<T> From<Array2<T>> for Mask<T> {
    fn from(bias: Array2<T>) -> Self {
        Self::Masked(bias)
    }
}

impl<T> From<Option<Array2<T>>> for Mask<T> {
    fn from(bias: Option<Array2<T>>) -> Self {
        match bias {
            Some(bias) => Self::Masked(bias),
            None => Self::Unmasked,
        }
    }
}

impl<T> From<Mask<T>> for Option<Array2<T>> {
    fn from(bias: Mask<T>) -> Self {
        match bias {
            Mask::Masked(bias) => Some(bias),
            Mask::Unmasked => None,
        }
    }
}

impl<T, D> ops::Add<Array<T, D>> for Mask<T>
where
    D: Dimension,
    T: NumOps + ScalarOperand,
    Array<T, D>: ops::Add<Array2<T>, Output = Array<T, D>>,
{
    type Output = Array<T, D>;

    fn add(self, rhs: Array<T, D>) -> Self::Output {
        use Mask::*;
        if let Masked(bias) = self {
            return rhs.clone() + bias;
        }
        rhs.clone()
    }
}

impl<T, D> ops::Add<&Array<T, D>> for Mask<T>
where
    D: Dimension,
    T: NumOps + ScalarOperand,
    Array<T, D>: ops::Add<Array2<T>, Output = Array<T, D>>,
{
    type Output = Array<T, D>;

    fn add(self, rhs: &Array<T, D>) -> Self::Output {
        use Mask::*;
        if let Masked(bias) = self {
            return rhs.clone() + bias;
        }
        rhs.clone()
    }
}

impl<T, D> ops::Add<Mask<T>> for Array<T, D>
where
    D: Dimension,
    T: NumOps + ScalarOperand,
    Array<T, D>: ops::Add<Array2<T>, Output = Array<T, D>>,
{
    type Output = Array<T, D>;

    fn add(self, bias: Mask<T>) -> Self::Output {
        use Mask::*;
        if let Masked(bias) = bias {
            return self.clone() + bias;
        }
        self.clone()
    }
}

impl<T, D> ops::Add<&Mask<T>> for Array<T, D>
where
    D: Dimension,
    T: NumOps + ScalarOperand,
    Array<T, D>: ops::Add<Array2<T>, Output = Array<T, D>>,
{
    type Output = Array<T, D>;

    fn add(self, bias: &Mask<T>) -> Self::Output {
        use Mask::*;
        if let Masked(m) = bias.clone() {
            return self.clone() + m;
        }
        self.clone()
    }
}
