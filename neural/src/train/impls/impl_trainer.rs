/*
    Appellation: impl_trainer <module>
    Contrib: @FL03
*/
use crate::model::Model;
use crate::train::trainer::Trainer;

impl<'a, M, T> core::ops::Deref for Trainer<'a, M, T>
where
    M: Model<T>,
{
    type Target = M;

    fn deref(&self) -> &Self::Target {
        self.model
    }
}
impl<'a, M, T> core::ops::DerefMut for Trainer<'a, M, T>
where
    M: Model<T>,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.model
    }
}
