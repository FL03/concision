/*
    Appellation: bias <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use ndarray::prelude::{Array, Array1};
use ndarray::Dimension;
use ndarray_rand::rand_distr::{uniform::SampleUniform, Uniform};
use ndarray_rand::RandomExt;
use num::Float;
use serde::{Deserialize, Serialize};
use smart_default::SmartDefault;
use strum::EnumIs;
use std::ops;

fn _generate_bias<T: Float + SampleUniform>(size: usize) -> Array1<T> {
    let ds = (T::from(size).unwrap()).sqrt();
    let dist = Uniform::new(-ds, ds);
    Array1::<T>::random(size, dist)
}

#[derive(Clone, Debug, Deserialize, EnumIs, PartialEq, Serialize, SmartDefault)]
pub enum Bias<T: Float = f64> {
    Biased(Array1<T>),
    #[default]
    Unbiased,
}

impl<T: Float> Bias<T> {
    pub fn forward(&self, data: &Array1<T>) -> Array1<T> {
        match self {
            Self::Biased(bias) => data + bias,
            Self::Unbiased => data.clone(),
        }
    }
}

impl<T: Float> Bias<T> where T: Float + SampleUniform {
    pub fn biased(size: usize) -> Self
    {
        let bias = _generate_bias(size);
        Self::Biased(bias)
    }
}

impl<D: Dimension, T: Float> ops::Add<Array<T, D>> for Bias<T>
where
    Array<T, D>: ops::Add<Array1<T>, Output = Array<T, D>>,
{
    type Output = Array<T, D>;

    fn add(self, rhs: Array<T, D>) -> Self::Output {
        match self {
            Self::Biased(bias) => rhs + bias,
            Self::Unbiased => rhs,
        }
    }
}

impl<D: Dimension, T: Float> ops::Add<&Array<T, D>> for Bias<T>
where
    Array<T, D>: ops::Add<Array1<T>, Output = Array<T, D>>,
{
    type Output = Array<T, D>;

    fn add(self, rhs: &Array<T, D>) -> Self::Output {
        match self {
            Self::Biased(bias) => rhs.clone() + bias,
            Self::Unbiased => rhs.clone(),
        }
    }
}

impl<D: Dimension, T: Float> ops::Add<Bias<T>> for Array<T, D>
where
    Array<T, D>: ops::Add<Array1<T>, Output = Array<T, D>>,
{
    type Output = Array<T, D>;

    fn add(self, bias: Bias<T>) -> Self::Output {
        match bias.clone() {
            Bias::Biased(bias) => self.clone() + bias,
            Bias::Unbiased => self.clone(),
        }
    }
}

impl<D: Dimension, T: Float> ops::Add<&Bias<T>> for Array<T, D>
where
    Array<T, D>: ops::Add<Array1<T>, Output = Array<T, D>>,
{
    type Output = Array<T, D>;

    fn add(self, bias: &Bias<T>) -> Self::Output {
        match bias.clone() {
            Bias::Biased(bias) => self.clone() + bias,
            Bias::Unbiased => self.clone(),
        }
    }
}
