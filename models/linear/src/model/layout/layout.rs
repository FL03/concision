/*
   Appellation: layout <module>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::model::layout::{features, Features};
use nd::{Dimension, RemoveAxis, ShapeBuilder};

#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]

pub struct Layout<D = nd::Ix2>
where
    D: Dimension,
{
    pub(crate) dim: D,
    pub(crate) features: Features,
}

impl<D> Layout<D>
where
    D: Dimension,
{
    pub fn new(dim: D) -> Self {
        let features = features(dim.clone()).expect("Invalid dimension");
        Self { dim, features }
    }

    pub fn from_shape<Sh>(shape: Sh) -> Self
    where
        D: RemoveAxis,
        Sh: ShapeBuilder<Dim = D>,
    {
        let shape = shape.into_shape();
        let dim = shape.raw_dim().clone();
        let features = Features::from_shape(shape);
        Self { dim, features }
    }

    pub fn as_slice(&self) -> &[usize] {
        self.dim.slice()
    }

    pub fn as_mut_slice(&mut self) -> &mut [usize] {
        self.dim.slice_mut()
    }

    pub fn features(&self) -> Features {
        self.features
    }

    pub fn ndim(&self) -> usize {
        self.dim.ndim()
    }

    pub fn pattern(&self) -> D::Pattern
    where
        D: Copy,
    {
        self.dim.into_pattern()
    }

    pub fn raw_dim(&self) -> D {
        self.dim.clone()
    }
}
