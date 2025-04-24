/*
    Appellation: impl_trainer <module>
    Contrib: @FL03
*/
use crate::model::Model;
use crate::train::trainer::Trainer;
use cnc::data::Records;

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
