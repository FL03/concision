/*
    appellation: impl_layer <module>
    authors: @FL03
*/
use crate::layers::LayerBase;

use crate::layers::{Activator, RawLayer};
use concision_params::{ParamsBase, RawParameter};
use concision_traits::Forward;
use ndarray::{DataOwned, Dimension, RawData, RemoveAxis, ShapeBuilder};

impl<F, S, D, A> LayerBase<F, ParamsBase<S, D, A>>
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

impl<F, P, X, Y> Forward<X> for LayerBase<F, P>
where
    F: Activator<Y, Output = Y>,
    P: RawParameter + Forward<X, Output = Y>,
{
    type Output = Y;

    fn forward(&self, input: &X) -> Self::Output {
        self.rho().activate(self.params().forward(input))
    }
}

impl<F, P, A> RawLayer<F, P> for LayerBase<F, P>
where
    F: Activator<P>,
    P: RawParameter<Elem = A>,
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
