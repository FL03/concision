/*
    Appellation: impl_trainer <module>
    Created At: 2025.11.28:13:12:11
    Contrib: @FL03
*/
use super::Trainer;
use crate::dataset::DatasetBase;
use crate::{IntoDataset, Records};
use concision_core::Model;

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
    pub const fn dataset(&self) -> &DatasetBase<R::Inputs, R::Targets> {
        &self.dataset
    }
    /// returns a mutable reference to the training session's dataset
    pub fn dataset_mut(&mut self) -> &mut DatasetBase<R::Inputs, R::Targets> {
        &mut self.dataset
    }

    pub fn begin(&self) -> &Self {
        todo!("Define a generic training loop...")
    }
}
