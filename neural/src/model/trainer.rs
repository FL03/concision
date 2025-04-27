/*
    Appellation: trainer <module>
    Contrib: @FL03
*/

use crate::Model;
use cnc::data::{Dataset, IntoDataset, Records};

pub struct Trainer<'a, M, T, R>
where
    M: Model<T>,
    R: Records,
{
    /// the training dataset
    pub(crate) dataset: Dataset<R::Inputs, R::Targets>,
    pub(crate) model: &'a mut M,
    /// the accumulated loss
    pub(crate) loss: T,
}

impl<'a, M, T, R> Trainer<'a, M, T, R>
where
    M: Model<T>,
    R: Records,
{
    pub fn new(model: &'a mut M, dataset: R) -> Self
    where
        R: IntoDataset<R::Inputs, R::Targets>,
        T: Default,
    {
        Self {
            dataset: dataset.into_dataset(),
            model,
            loss: T::default(),
        }
    }
    /// returns an immutable reference to the total loss
    pub const fn loss(&self) -> &T {
        &self.loss
    }
    /// returns a mutable reference to the total loss
    pub fn loss_mut(&mut self) -> &mut T {
        &mut self.loss
    }
    /// returns an immutable reference to the training session's dataset
    pub const fn dataset(&self) -> &Dataset<R::Inputs, R::Targets> {
        &self.dataset
    }
    /// returns a mutable reference to the training session's dataset
    pub fn dataset_mut(&mut self) -> &mut Dataset<R::Inputs, R::Targets> {
        &mut self.dataset
    }
    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self)))]
    pub fn begin(&self) -> &Self {
        todo!("Define a generic training loop...")
    }
}

impl<'a, M, T, R> core::ops::Deref for Trainer<'a, M, T, R>
where
    M: Model<T>,
    R: Records,
{
    type Target = M;

    fn deref(&self) -> &Self::Target {
        self.model
    }
}
impl<'a, M, T, R> core::ops::DerefMut for Trainer<'a, M, T, R>
where
    M: Model<T>,
    R: Records,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.model
    }
}
