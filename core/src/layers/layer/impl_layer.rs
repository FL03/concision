/*
    appellation: impl_layer <module>
    authors: @FL03
*/
use crate::layers::LayerBase;

use crate::layers::{Activator, Layer};
use concision_params::ParamsBase;
use ndarray::{DataOwned, Dimension, RawData, RemoveAxis, ShapeBuilder};

impl<A, F, S, D> LayerBase<F, ParamsBase<S, D, A>>
where
    F: Activator<A, Output = A>,
    D: Dimension,
    S: RawData<Elem = A>,
{
    /// create a new [`LayerBase`] from the given activation function and shape.
    pub fn from_rho_with_shape<Sh>(rho: F, shape: Sh) -> Self
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
}

impl<A, F, S, D> Layer<S, D> for LayerBase<F, ParamsBase<S, D, A>>
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

    fn params(&self) -> &ParamsBase<S, D, A> {
        &self.params
    }

    fn params_mut(&mut self) -> &mut ParamsBase<S, D, A> {
        &mut self.params
    }
}
