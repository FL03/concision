/*
    appellation: impl_layer <module>
    authors: @FL03
*/
use crate::layers::LayerBase;

use crate::layers::{Activator, RawLayer};
use concision_params::{ParamsBase, RawParameter};
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

impl<F, P, A> RawLayer<F, P> for LayerBase<F, P>
where
    F: Activator<P>,
    P: RawParameter<Elem = A>,
{
    type Elem = A;

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
