/*
    Appellation: layer <module>
    Contrib: @FL03
*/
#![allow(unused)]
use crate::ActivateGradient;

use super::{Activate, Layer};
use cnc::{Forward, ParamsBase, activate};
use ndarray::{Dimension, Ix2, RawData};

pub struct LayerBase<F, S, D = Ix2>
where
    D: Dimension,
    S: RawData,
{
    pub(crate) rho: F,
    pub(crate) params: ParamsBase<S, D>,
}

pub struct Linear;

impl<U> Activate<U> for Linear {
    type Output = U;

    fn activate(&self, x: U) -> Self::Output {
        x
    }
}

impl<S, D> LayerBase<Linear, S, D>
where
    D: Dimension,
    S: RawData<Elem = f32>,
{
    pub fn linear(params: ParamsBase<S, D>) -> Self {
        Self {
            rho: Linear,
            params,
        }
    }
}

impl<F, S, A, D> LayerBase<F, S, D>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    pub fn new(rho: F, params: ParamsBase<S, D>) -> Self {
        Self { rho, params }
    }
    /// returns an immutable reference to the layer's parameters
    pub fn params(&self) -> &ParamsBase<S, D> {
        &self.params
    }
    /// returns a mutable reference to the layer's parameters
    pub fn params_mut(&mut self) -> &mut ParamsBase<S, D> {
        &mut self.params
    }
    /// returns an immutable reference to the activation function of the layer
    pub fn rho(&self) -> &F {
        &self.rho
    }
    /// returns a mutable reference to the activation function of the layer
    pub fn rho_mut(&mut self) -> &mut F {
        &mut self.rho
    }
    /// consumes the current instance and returns another with the given activation function.
    /// This is useful during the creation of the model, when the activation function is not known yet.
    pub fn with_rho<G>(self, rho: G) -> LayerBase<G, S, D>
    where
        G: Activate<S::Elem>,
        F: Activate<S::Elem>,
        S: RawData<Elem = A>,
    {
        LayerBase {
            rho,
            params: self.params,
        }
    }
    pub fn forward<X, Y>(&self, input: &X) -> cnc::Result<Y>
    where
        F: Activate<<ParamsBase<S, D> as Forward<X>>::Output, Output = Y>,
        ParamsBase<S, D>: Forward<X, Output = Y>,
        X: Clone,
        Y: Clone,
    {
        Forward::forward(&self.params, input).map(|x| self.rho.activate(x))
    }
}

impl<F, S, D> core::ops::Deref for LayerBase<F, S, D>
where
    D: Dimension,
    S: RawData,
{
    type Target = ParamsBase<S, D>;

    fn deref(&self) -> &Self::Target {
        &self.params
    }
}

impl<F, S, D> core::ops::DerefMut for LayerBase<F, S, D>
where
    D: Dimension,
    S: RawData,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.params
    }
}

impl<F, H, S, D> Activate<H> for LayerBase<F, S, D>
where
    F: Activate<H>,
    D: Dimension,
    S: RawData,
{
    type Output = F::Output;

    fn activate(&self, x: H) -> Self::Output {
        self.rho.activate(x)
    }
}

impl<A, F, S, D> ActivateGradient<A> for LayerBase<F, S, D>
where
    F: ActivateGradient<A>,
    D: Dimension,
    S: RawData<Elem = A>,
{
    type Input = F::Input;
    type Delta = F::Delta;

    fn activate_gradient(&self, x: &A) -> Self::Delta {
        self.rho.activate_gradient(x)
    }
}

impl<A, B, S, D> super::Layer<S, D> for LayerBase<Box<dyn Activate<A, Output = B> + 'static>, S, D>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    type Scalar = A;
    type Rho<U> = Box<dyn Activate<U, Output = U> + 'static>;

    fn params(&self) -> &ParamsBase<S, D> {
        &self.params
    }

    fn params_mut(&mut self) -> &mut ParamsBase<S, D> {
        &mut self.params
    }
}
