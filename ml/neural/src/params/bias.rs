/*
    Appellation: bias <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::generate_uniform_arr;
use ndarray::prelude::{Array, Array1};
use ndarray::{Dimension, ScalarOperand};
use ndarray_rand::rand_distr::uniform::SampleUniform;
use num::Float;
use serde::{Deserialize, Serialize};
use smart_default::SmartDefault;
use std::ops;
use strum::EnumIs;

pub struct Belief<T: Float = f64> {
    pub bias: Bias<T>,
    pub features: usize,
}

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
    T: Float + ScalarOperand,
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
    T: Float + ScalarOperand,
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
    T: Float + ScalarOperand,
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
    T: Float + ScalarOperand,
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
    T: Float + ScalarOperand,
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
    T: Float + ScalarOperand,
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
    T: Float + ScalarOperand,
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
    T: Float + ScalarOperand,
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
    T: Float + ScalarOperand,
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
    T: Float + ScalarOperand,
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
