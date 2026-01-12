/*
    Appellation: impl_layer_ext <module>
    Created At: 2026.01.12:09:33:36
    Contrib: @FL03
*/
use crate::nn::layer::LayerBase;
use crate::nn::{RawLayer, RawLayerMut};
use concision_params::RawParams;
use concision_traits::{Activator, Forward};

impl<F, P, A, X, Y> Activator<X> for LayerBase<F, P>
where
    F: Activator<X, Output = Y>,
    P: RawParams<Elem = A>,
{
    type Output = F::Output;

    fn activate(&self, input: X) -> Self::Output {
        self.rho().activate(input)
    }
}

impl<F, P, A, X, Y, Z> Forward<X> for LayerBase<F, P>
where
    F: Activator<Y, Output = Z>,
    P: RawParams<Elem = A> + Forward<X, Output = Y>,
{
    type Output = F::Output;

    fn forward(&self, input: &X) -> Self::Output {
        self.rho().activate(self.params().forward(input))
    }
}

impl<F, P, A> RawLayer<F, A> for LayerBase<F, P>
where
    F: Activator<A>,
    P: RawParams<Elem = A>,
{
    type Params<_T> = P;

    fn rho(&self) -> &F {
        &self.rho
    }

    fn params(&self) -> &P {
        &self.params
    }
}
impl<F, P, A> RawLayerMut<F, A> for LayerBase<F, P>
where
    F: Activator<A>,
    P: RawParams<Elem = A>,
{
    fn params_mut(&mut self) -> &mut P {
        &mut self.params
    }
}
