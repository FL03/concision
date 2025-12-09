/*
    Appellation: impl_model_layout <module>
    Created At: 2025.12.09:07:46:57
    Contrib: @FL03
*/
use super::ModelLayout;

use crate::layout::{Deep, NetworkDepth, RawModelLayout, Shallow};

impl<F, D> ModelLayout<F, D>
where
    F: RawModelLayout,
    D: NetworkDepth,
{
    /// creates a new instance of [`ModelLayout`] using the given features
    pub const fn new(features: F) -> Self {
        Self {
            features,
            _marker: core::marker::PhantomData::<D>,
        }
    }
    /// returns a reference to the features of the model layout
    pub const fn features(&self) -> &F {
        &self.features
    }
    /// returns a mutable reference to the features of the model layout
    pub const fn features_mut(&mut self) -> &mut F {
        &mut self.features
    }
    /// returns a reference to the input of the model layout
    pub fn input(&self) -> usize {
        self.features().input()
    }
    /// returns a reference to the output of the model layout
    pub fn output(&self) -> usize {
        self.features().output()
    }
    /// returns a reference to the hidden features of the model layout
    pub fn hidden(&self) -> usize {
        self.features().hidden()
    }
    /// returns a reference to the depth, or number of hidden layers, of the model
    pub fn layers(&self) -> usize {
        self.features().depth()
    }
}

impl<F, D> core::ops::Deref for ModelLayout<F, D>
where
    F: RawModelLayout,
    D: NetworkDepth,
{
    type Target = F;

    fn deref(&self) -> &Self::Target {
        &self.features
    }
}

impl<F, D> core::ops::DerefMut for ModelLayout<F, D>
where
    F: RawModelLayout,
    D: NetworkDepth,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.features
    }
}

impl<F> ModelLayout<F, Deep>
where
    F: RawModelLayout,
{
    /// creates a new instance of [`ModelLayout`] using the given features
    pub const fn deep(features: F) -> Self {
        Self::new(features)
    }
}

impl<F> ModelLayout<F, Shallow>
where
    F: RawModelLayout,
{
    /// returns a new instance of the model layout using the given features and a shallow depth
    pub const fn shallow(features: F) -> Self {
        Self::new(features)
    }
}
