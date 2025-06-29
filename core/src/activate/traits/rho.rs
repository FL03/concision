
/// The [Activate] trait enables the definition of new activation functions often implemented
/// as _fieldless_ structs.
pub trait Activate<Rhs = Self> {
    type Output;

    fn activate(&self, rhs: Rhs) -> Self::Output;
}

pub trait ActivateGradient<Rhs = Self>: Activate<Self::Input> {
    type Input;
    type Delta;

    fn activate_gradient(&self, rhs: Rhs) -> Self::Delta;
}

/*
 ************* Implementations *************
*/

impl<X, Y> Activate<X> for Box<dyn Activate<X, Output = Y>> {
    type Output = Y;

    fn activate(&self, rhs: X) -> Self::Output {
        self.as_ref().activate(rhs)
    }
}

impl<X, Y, F> Activate<X> for F
where
    F: Fn(X) -> Y,
{
    type Output = Y;

    fn activate(&self, rhs: X) -> Self::Output {
        self(rhs)
    }
}
