/*
    Appellation: bias <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::core::prelude::GenerateRandom;
use crate::generate_uniform_arr;
use ndarray::prelude::{Array, Array1, Dimension, Ix1, NdFloat};
use ndarray::IntoDimension;
use ndarray_rand::rand_distr::uniform::SampleUniform;
use num::Float;
use serde::{Deserialize, Serialize};
use smart_default::SmartDefault;
use std::ops;
use strum::EnumIs;

pub struct Belief<T = f64, D = Ix1>
where
    D: Dimension,
    T: Float,
{
    pub bias: Array<T, D>,
    pub features: D,
}

#[derive(Clone, Debug, Deserialize, EnumIs, PartialEq, Serialize, SmartDefault)]
pub enum Biases<T = f64, D = Ix1>
where
    D: Dimension,
    T: Float,
{
    Biased(Array<T, D>),
    #[default]
    Unbiased,
}

impl<T, D> Biases<T, D>
where
    D: Dimension,
    T: Float,
{
    pub fn biased(bias: Array<T, D>) -> Self {
        Self::Biased(bias)
    }
}

impl<T, D> Biases<T, D>
where
    D: Dimension,
    T: Float + SampleUniform,
{
    pub fn init(self, dk: T, features: impl IntoDimension<Dim = D>) -> Self {
        Self::Biased(Array::uniform_between(dk, features))
    }
    pub fn uniform(dk: T, features: impl IntoDimension<Dim = D>) -> Self {
        Self::Biased(Array::uniform_between(dk, features))
    }
}

impl<T, D> From<Array<T, D>> for Biases<T, D>
where
    D: Dimension,
    T: Float,
{
    fn from(bias: Array<T, D>) -> Self {
        Self::Biased(bias)
    }
}

impl<T, D> From<Option<Array<T, D>>> for Biases<T, D>
where
    D: Dimension,
    T: Float,
{
    fn from(bias: Option<Array<T, D>>) -> Self {
        match bias {
            Some(bias) => Self::Biased(bias),
            None => Self::Unbiased,
        }
    }
}

impl<T, D> From<Biases<T, D>> for Option<Array<T, D>>
where
    D: Dimension,
    T: Float,
{
    fn from(bias: Biases<T, D>) -> Self {
        match bias {
            Biases::Biased(bias) => Some(bias),
            Biases::Unbiased => None,
        }
    }
}

impl<T, D> From<Biases<T, D>> for Array<T, D>
where
    D: Dimension,
    T: Float,
{
    fn from(bias: Biases<T, D>) -> Self {
        match bias {
            Biases::Biased(bias) => bias,
            Biases::Unbiased => Array::zeros(D::zeros(D::NDIM.unwrap_or_default())),
        }
    }
}

// impl<'a, T, D> From<&'a Biases<T, D>> for ArrayView<'a, T, D>
// where
//     D: Dimension + 'a,
//     T: Float,
// {
//     fn from(bias: &'a Biases<T, D>) -> Self {
//         match bias {
//             Biases::Biased(bias) => bias.view(),
//             Biases::Unbiased => ArrayView::empty(D::zeros(D::NDIM.unwrap_or_default())),
//         }
//     }
// }

#[derive(Clone, Debug, Deserialize, EnumIs, PartialEq, Serialize, SmartDefault)]
pub enum Bias<T: Float = f64> {
    Biased(Array1<T>),
    #[default]
    Unbiased,
    Value(T),
}

impl<T> Bias<T>
where
    T: Float,
{
    pub fn update_at(&mut self, index: usize, value: T) {
        match self {
            Self::Biased(bias) => bias[index] = value,
            Self::Unbiased => (),
            Self::Value(bias) => *bias = value,
        }
    }
}

impl<T> Bias<T>
where
    T: NdFloat,
{
    pub fn forward(&self, data: &Array1<T>) -> Array1<T> {
        match self {
            Self::Biased(bias) => data + bias,
            Self::Unbiased => data.clone(),
            Self::Value(value) => data + value.clone(),
        }
    }
}

impl<T> Bias<T>
where
    T: Float + SampleUniform,
{
    pub fn biased(size: usize) -> Self {
        let bias = generate_uniform_arr(0, size);
        Self::Biased(bias)
    }
}

impl<T> From<T> for Bias<T>
where
    T: Float,
{
    fn from(value: T) -> Self {
        Self::Value(value)
    }
}

impl<T> From<Array1<T>> for Bias<T>
where
    T: Float,
{
    fn from(bias: Array1<T>) -> Self {
        Self::Biased(bias)
    }
}

impl<T> From<Option<Array1<T>>> for Bias<T>
where
    T: Float,
{
    fn from(bias: Option<Array1<T>>) -> Self {
        match bias {
            Some(bias) => Self::Biased(bias),
            None => Self::Unbiased,
        }
    }
}

impl<T> From<Bias<T>> for Option<Array1<T>>
where
    T: Float,
{
    fn from(bias: Bias<T>) -> Self {
        match bias {
            Bias::Biased(bias) => Some(bias),
            Bias::Unbiased => None,
            Bias::Value(value) => Some(Array1::<T>::from_elem(1, value)),
        }
    }
}

impl<T, D> ops::Add<Array<T, D>> for Bias<T>
where
    D: Dimension,
    T: NdFloat,
    Array<T, D>: ops::Add<Array1<T>, Output = Array<T, D>>,
{
    type Output = Array<T, D>;

    fn add(self, rhs: Array<T, D>) -> Self::Output {
        match self {
            Self::Biased(bias) => rhs + bias,
            Self::Unbiased => rhs,
            Self::Value(value) => &rhs + value,
        }
    }
}

impl<T, D> ops::Add<Array<T, D>> for &Bias<T>
where
    D: Dimension,
    T: NdFloat,
    Array<T, D>: ops::Add<Array1<T>, Output = Array<T, D>>,
{
    type Output = Array<T, D>;

    fn add(self, rhs: Array<T, D>) -> Self::Output {
        match self.clone() {
            Bias::Biased(bias) => rhs + bias,
            Bias::Unbiased => rhs,
            Bias::Value(value) => &rhs + value,
        }
    }
}

impl<T, D> ops::Add<&Array<T, D>> for Bias<T>
where
    D: Dimension,
    T: NdFloat,
    Array<T, D>: ops::Add<Array1<T>, Output = Array<T, D>>,
{
    type Output = Array<T, D>;

    fn add(self, rhs: &Array<T, D>) -> Self::Output {
        match self {
            Self::Biased(bias) => rhs.clone() + bias,
            Self::Unbiased => rhs.clone(),
            Self::Value(value) => rhs + value,
        }
    }
}

impl<T, D> ops::Add<Bias<T>> for Array<T, D>
where
    D: Dimension,
    T: NdFloat,
    Array<T, D>: ops::Add<Array1<T>, Output = Array<T, D>>,
{
    type Output = Array<T, D>;

    fn add(self, bias: Bias<T>) -> Self::Output {
        match bias.clone() {
            Bias::Biased(bias) => self.clone() + bias,
            Bias::Unbiased => self.clone(),
            Bias::Value(value) => &self + value,
        }
    }
}

impl<T, D> ops::Add<&Bias<T>> for Array<T, D>
where
    D: Dimension,
    T: NdFloat,
    Array<T, D>: ops::Add<Array1<T>, Output = Array<T, D>>,
{
    type Output = Array<T, D>;

    fn add(self, bias: &Bias<T>) -> Self::Output {
        match bias.clone() {
            Bias::Biased(bias) => self.clone() + bias,
            Bias::Unbiased => self.clone(),
            Bias::Value(value) => &self + value,
        }
    }
}

impl<T, D> ops::Sub<Array<T, D>> for Bias<T>
where
    D: Dimension,
    T: NdFloat,
    Array<T, D>: ops::Sub<Array1<T>, Output = Array<T, D>>,
{
    type Output = Array<T, D>;

    fn sub(self, rhs: Array<T, D>) -> Self::Output {
        match self {
            Self::Biased(bias) => rhs - bias,
            Self::Unbiased => rhs,
            Self::Value(value) => &rhs - value,
        }
    }
}

impl<T, D> ops::Sub<&Array<T, D>> for Bias<T>
where
    D: Dimension,
    T: NdFloat,
    Array<T, D>: ops::Sub<Array1<T>, Output = Array<T, D>>,
{
    type Output = Array<T, D>;

    fn sub(self, rhs: &Array<T, D>) -> Self::Output {
        match self {
            Self::Biased(bias) => rhs.clone() - bias,
            Self::Unbiased => rhs.clone(),
            Self::Value(value) => rhs - value,
        }
    }
}

impl<T, D> ops::Sub<Bias<T>> for Array<T, D>
where
    D: Dimension,
    T: NdFloat,
    Array<T, D>: ops::Sub<Array1<T>, Output = Array<T, D>>,
{
    type Output = Array<T, D>;

    fn sub(self, bias: Bias<T>) -> Self::Output {
        match bias.clone() {
            Bias::Biased(bias) => self.clone() - bias,
            Bias::Unbiased => self.clone(),
            Bias::Value(value) => &self - value,
        }
    }
}

impl<T, D> ops::Sub<&Bias<T>> for Array<T, D>
where
    D: Dimension,
    T: NdFloat,
    Array<T, D>: ops::Sub<Array1<T>, Output = Array<T, D>>,
{
    type Output = Array<T, D>;

    fn sub(self, bias: &Bias<T>) -> Self::Output {
        match bias.clone() {
            Bias::Biased(bias) => self.clone() - bias,
            Bias::Unbiased => self.clone(),
            Bias::Value(value) => &self - value,
        }
    }
}
