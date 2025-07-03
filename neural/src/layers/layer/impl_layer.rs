/*
    appellation: impl_layer <module>
    authors: @FL03
*/
use crate::layers::LayerBase;

use crate::layers::{Activator, ActivatorGradient, Layer};
use cnc::{Forward, ParamsBase};
use ndarray::{Data, Dimension, RawData};

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

impl<A, X, Y, F, S, D> Forward<X> for LayerBase<F, S, D>
where
    A: Clone,
    F: Activator<Y, Output = Y>,
    D: Dimension,
    S: Data<Elem = A>,
    ParamsBase<S, D>: Forward<X, Output = Y>,
{
    type Output = Y;

    fn forward(&self, inputs: &X) -> cnc::Result<Self::Output> {
        let y = self.params().forward(inputs)?;

        Ok(self.rho().activate(y))
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
    F: Activator<A, Output = A>,
    D: Dimension,
    S: RawData<Elem = A>,
{
    type Elem = A;
    type Rho = F;

    fn rho(&self) -> &Self::Rho {
        &self.rho
    }

    fn params(&self) -> &ParamsBase<S, D> {
        &self.params
    }

    fn params_mut(&mut self) -> &mut ParamsBase<S, D> {
        &mut self.params
    }
}
