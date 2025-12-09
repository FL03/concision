/*
    Appellation: unary <module>
    Created At: 2025.12.09:07:26:04
    Contrib: @FL03
*/
/// The [`Decrement`] trait defines a method enabling an axis to decrement itself,
pub trait Decrement {
    type Output;

    fn dec(self) -> Self::Output;
}
/// The [`Increment`] trait defines a method enabling an axis to increment itself,
/// effectively adding a new axis to the array.
pub trait Increment {
    type Output;

    fn inc(self) -> Self::Output;
}

/*
    ************* Implementations *************
*/
use ndarray::{Axis, Dimension, RemoveAxis,};

impl<D, E> Decrement for D
where
    D: RemoveAxis<Smaller = E>,
    E: Dimension,
{
    type Output = E;

    fn dec(self) -> Self::Output {
        self.remove_axis(Axis(self.ndim() - 1))
    }
}

impl<D, E> Increment for D
where
    D: Dimension<Larger = E>,
    E: Dimension,
{
    type Output = E;

    fn inc(self) -> Self::Output {
        self.insert_axis(Axis(self.ndim()))
    }
}