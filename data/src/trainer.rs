/*
    Appellation: trainer <module>
    Contrib: @FL03
*/

mod impl_trainer;

use crate::Records;
use crate::dataset::DatasetBase;
use concision_core::Model;

pub trait ModelTrainer<T> {
    type Model: Model<T>;
    /// returns a model trainer prepared to train the model; this is a convenience method
    /// that creates a new trainer instance and returns it. Trainers are lazily evaluated
    /// meaning that the training process won't begin until the user calls the `begin` method.
    fn trainer<'a, U, V>(
        &mut self,
        dataset: DatasetBase<U, V>,
        model: &'a mut Self::Model,
    ) -> Trainer<'a, Self::Model, T, DatasetBase<U, V>>
    where
        Self: Sized,
        T: Default,
        for<'b> &'b mut Self::Model: Model<T>,
    {
        Trainer::new(model, dataset)
    }
}

/// The [`Trainer`] is a generalized model trainer that works to provide a common interface for
/// training models over datasets.
pub struct Trainer<'a, M, T, R>
where
    M: Model<T>,
    R: Records,
{
    /// the training dataset
    pub(crate) dataset: DatasetBase<R::Inputs, R::Targets>,
    pub(crate) model: &'a mut M,
    /// the accumulated loss
    pub(crate) loss: T,
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
