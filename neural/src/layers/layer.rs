/*
    Appellation: layer <module>
    Contrib: @FL03
*/
mod impl_layer_repr;

use super::{Activator, ActivatorGradient, Layer};
use cnc::{Forward, ParamsBase};
use ndarray::{DataOwned, Dimension, Ix2, RawData, RemoveAxis, ShapeBuilder};

/// The [`LayerBase`] struct is a base representation of a neural network layer, essentially
/// binding an activation function, `F`, to a set of parameters, `ParamsBase<S, D>`.
pub struct LayerBase<F, S, D = Ix2>
where
    D: Dimension,
    S: RawData,
{
    /// the activation function of the layer
    pub(crate) rho: F,
    /// the parameters of the layer is an object consisting of both a weight and a bias tensor.
    pub(crate) params: ParamsBase<S, D>,
}

impl<F, S, A, D> LayerBase<F, S, D>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    /// create a new [`LayerBase`] from the given activation function and parameters.
    pub const fn new(rho: F, params: ParamsBase<S, D>) -> Self {
        Self { rho, params }
    }
    /// create a new [`LayerBase`] from the given parameters assuming the logical default for
    /// the activation of type `F`.
    pub fn from_params(params: ParamsBase<S, D>) -> Self
    where
        F: Default,
    {
        Self {
            rho: F::default(),
            params,
        }
    }
    /// create a new [`LayerBase`] from the given activation function and shape.
    pub fn from_rho<Sh>(rho: F, shape: Sh) -> Self
    where
        A: Clone + Default,
        S: DataOwned,
        D: RemoveAxis,
        Sh: ShapeBuilder<Dim = D>,
    {
        Self {
            rho,
            params: ParamsBase::default(shape),
        }
    }
    /// returns an immutable reference to the layer's parameters
    pub const fn params(&self) -> &ParamsBase<S, D> {
        &self.params
    }
    /// returns a mutable reference to the layer's parameters
    pub const fn params_mut(&mut self) -> &mut ParamsBase<S, D> {
        &mut self.params
    }
    /// returns an immutable reference to the activation function of the layer
    pub const fn rho(&self) -> &F {
        &self.rho
    }
    /// returns a mutable reference to the activation function of the layer
    pub const fn rho_mut(&mut self) -> &mut F {
        &mut self.rho
    }
    /// consumes the current instance and returns another with the given parameters.
    pub fn with_params<S2, D2>(self, params: ParamsBase<S2, D2>) -> LayerBase<F, S2, D2>
    where
        S2: RawData<Elem = S::Elem>,
        D2: Dimension,
    {
        LayerBase {
            rho: self.rho,
            params,
        }
    }
    /// consumes the current instance and returns another with the given activation function.
    /// This is useful during the creation of the model, when the activation function is not known yet.
    pub fn with_rho<G>(self, rho: G) -> LayerBase<G, S, D>
    where
        G: Activator<S::Elem>,
        F: Activator<S::Elem>,
        S: RawData<Elem = A>,
    {
        LayerBase {
            rho,
            params: self.params,
        }
    }
    pub fn forward<X, Y>(&self, input: &X) -> cnc::Result<Y>
    where
        F: Activator<<ParamsBase<S, D> as Forward<X>>::Output, Output = Y>,
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

impl<U, V, F, S, D> Activator<U> for LayerBase<F, S, D>
where
    F: Activator<U, Output = V>,
    D: Dimension,
    S: RawData,
{
    type Output = V;

    fn activate(&self, x: U) -> Self::Output {
        self.rho().activate(x)
    }
}

impl<U, F, S, D> ActivatorGradient<U> for LayerBase<F, S, D>
where
    F: ActivatorGradient<U>,
    D: Dimension,
    S: RawData,
{
    type Input = F::Input;
    type Delta = F::Delta;

    fn activate_gradient(&self, inputs: F::Input) -> F::Delta {
        self.rho().activate_gradient(inputs)
    }
}

impl<A, F, S, D> Layer<S, D> for LayerBase<F, S, D>
where
    F: ActivatorGradient<A>,
    D: Dimension,
    S: RawData<Elem = A>,
{
    type Scalar = A;

    fn params(&self) -> &ParamsBase<S, D> {
        &self.params
    }

    fn params_mut(&mut self) -> &mut ParamsBase<S, D> {
        &mut self.params
    }
}
