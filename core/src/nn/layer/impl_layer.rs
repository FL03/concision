/*
    appellation: impl_layer <module>
    authors: @FL03
*/
use super::Layer;
use crate::activate::Activator;
use crate::nn::RawLayer;
use concision_params::RawParams;
use concision_traits::Forward;

impl<F, P, A> Layer<F, P>
where
    P: RawParams<Elem = A>,
{
    /// create a new [`Layer`] from the given activation function and parameters.
    pub const fn new(rho: F, params: P) -> Self {
        Self { rho, params }
    }
    /// create a new [`Layer`] from the given parameters assuming the logical default for
    /// the activation of type `F`.
    pub fn from_params(params: P) -> Self
    where
        F: Default,
    {
        Self::new(<F>::default(), params)
    }
    /// create a new [`Layer`] from the given activation function and shape.
    pub fn from_rho<Sh>(rho: F) -> Self
    where
        P: Default,
    {
        Self::new(rho, <P>::default())
    }
    /// returns an immutable reference to the layer's parameters
    pub const fn params(&self) -> &P {
        &self.params
    }
    /// returns a mutable reference to the layer's parameters
    pub const fn params_mut(&mut self) -> &mut P {
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
    pub fn with_params<Y>(self, params: Y) -> Layer<F, Y>
    where
        F: Activator<Y>,
    {
        Layer {
            rho: self.rho,
            params,
        }
    }
    /// consumes the current instance and returns another with the given activation function.
    /// This is useful during the creation of the model, when the activation function is not known yet.
    pub fn with_rho<G>(self, rho: G) -> Layer<G, P>
    where
        G: Activator<P>,
        F: Activator<P>,
    {
        Layer {
            rho,
            params: self.params,
        }
    }
    /// given some input, complete a single forward pass through the layer
    pub fn forward<U, V>(&self, input: &U) -> V
    where
        Self: Forward<U, Output = V>,
    {
        <Self as Forward<U>>::forward(self, input)
    }
}

impl<F, P, X, Y> Forward<X> for Layer<F, P>
where
    F: Activator<Y, Output = Y>,
    P: Forward<X, Output = Y>,
{
    type Output = Y;

    fn forward(&self, input: &X) -> Self::Output {
        self.params.forward_then(input, |y| self.rho.activate(y))
    }
}

impl<F, P, A> RawLayer<F, P> for Layer<F, P>
where
    F: Activator<A>,
    P: RawParams<Elem = A>,
{
    fn rho(&self) -> &F {
        &self.rho
    }

    fn params(&self) -> &P {
        &self.params
    }

    fn params_mut(&mut self) -> &mut P {
        &mut self.params
    }
}
