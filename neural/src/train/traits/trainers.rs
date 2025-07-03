/*
    appellation: trainers <module>
    authors: @FL03
*/
use crate::train::Trainer;

use crate::Model;
use concision_data::DatasetBase;

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
