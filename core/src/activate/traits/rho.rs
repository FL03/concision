/*
    appellation: rho <module>
    authors: @FL03
*/
/// The [`Rho`] trait enables the definition of new activation functions often implemented
/// as _fieldless_ structs.
pub trait Rho<Rhs = Self> {
    type Output;

    fn rho(&self, rhs: Rhs) -> Self::Output;
}

pub trait RhoGradient<Rhs = Self>: Rho<Self::Input> {
    type Input;
    type Delta;

    fn rho_gradient(&self, rhs: Rhs) -> Self::Delta;
}

/*
 ************* Implementations *************
*/
#[cfg(feature = "alloc")]
use alloc::boxed::Box;

#[cfg(feature = "alloc")]
impl<X, Y> Rho<X> for Box<dyn Rho<X, Output = Y>> {
    type Output = Y;

    fn rho(&self, rhs: X) -> Self::Output {
        self.as_ref().rho(rhs)
    }
}

impl<X, Y, F> Rho<X> for F
where
    F: Fn(X) -> Y,
{
    type Output = Y;

    fn rho(&self, rhs: X) -> Self::Output {
        self(rhs)
    }
}
