/*
    Appellation: unary <module>
    Created At: 2025.12.09:07:26:04
    Contrib: @FL03
*/
/// [`Decrement`] is a chainable trait that defines a decrement method,
/// effectively removing a single unit from the original object to create another
pub trait Decrement {
    type Output;

    fn dec(self) -> Self::Output;
}

/// The [`DecrementMut`] trait defines a decrement method that operates in place,
/// modifying the original object.
pub trait DecrementMut {
    fn dec_mut(&mut self);
}
/// The [`Increment`]
pub trait Increment {
    type Output;

    fn inc(self) -> Self::Output;
}

pub trait IncrementMut {
    fn inc_mut(&mut self);
}

/*
 ************* Implementations *************
*/
use num_traits::One;

impl<T> Decrement for T
where
    T: One + core::ops::Sub<Output = T>,
{
    type Output = T;

    fn dec(self) -> Self::Output {
        self - T::one()
    }
}

impl<T> DecrementMut for T
where
    T: One + core::ops::SubAssign,
{
    fn dec_mut(&mut self) {
        *self -= T::one()
    }
}

impl<T> Increment for T
where
    T: One + core::ops::Add<Output = T>,
{
    type Output = T;

    fn inc(self) -> Self::Output {
        self + T::one()
    }
}
