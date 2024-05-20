/*
    Appellation: xavier <distr>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Xavier
//!
//! Xavier initialization techniques were developed in 2010 by Xavier Glorot.
//! These methods are designed to initialize the weights of a neural network in a way that
//! prevents the vanishing and exploding gradient problems. The initialization technique
//! manifests into two distributions: [XavierNormal] and [XavierUniform].
// #76
use num::Float;
use rand::Rng;
use rand_distr::uniform::{SampleUniform, Uniform};
use rand_distr::{Distribution, Normal, NormalError, StandardNormal};

pub(crate) fn std_dev<F>(inputs: usize, outputs: usize) -> F
where
    F: Float,
{
    (F::from(2).unwrap() / F::from(inputs + outputs).unwrap()).sqrt()
}

pub(crate) fn boundary<F>(inputs: usize, outputs: usize) -> F
where
    F: Float,
{
    (F::from(6).unwrap() / F::from(inputs + outputs).unwrap()).sqrt()
}
/// Normal Xavier initializers leverage a normal distribution with a mean of 0 and a standard deviation (`σ`)
/// computed by the formula: `σ = sqrt(2/(d_in + d_out))`
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct XavierNormal<F>
where
    F: Float,
    StandardNormal: Distribution<F>,
{
    std: F,
}

impl<F> XavierNormal<F>
where
    F: Float,
    StandardNormal: Distribution<F>,
{
    pub fn new(inputs: usize, outputs: usize) -> Self {
        Self {
            std: std_dev(inputs, outputs),
        }
    }

    pub fn distr(&self) -> Result<Normal<F>, NormalError> {
        Normal::new(F::zero(), self.std_dev())
    }

    pub fn std_dev(&self) -> F {
        self.std
    }
}

impl<F> Distribution<F> for XavierNormal<F>
where
    F: Float,
    StandardNormal: Distribution<F>,
{
    fn sample<R>(&self, rng: &mut R) -> F
    where
        R: Rng + ?Sized,
    {
        self.distr().unwrap().sample(rng)
    }
}

/// Uniform Xavier initializers use a uniform distribution to initialize the weights of a neural network
/// within a given range.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct XavierUniform<X>
where
    X: SampleUniform,
{
    boundary: X,
}

impl<X> XavierUniform<X>
where
    X: Float + SampleUniform,
{
    pub fn new(inputs: usize, outputs: usize) -> Self {
        Self {
            boundary: boundary(inputs, outputs),
        }
    }

    pub fn boundary(&self) -> X {
        self.boundary
    }

    pub fn distr(&self) -> Uniform<X>
    where
        X: Float,
    {
        let bnd = self.boundary();
        Uniform::new(-bnd, bnd)
    }
}

impl<X> Distribution<X> for XavierUniform<X>
where
    X: Float + SampleUniform,
{
    fn sample<R>(&self, rng: &mut R) -> X
    where
        R: Rng + ?Sized,
    {
        self.distr().sample(rng)
    }
}
