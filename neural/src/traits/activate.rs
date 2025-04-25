pub trait BinaryAction<A, B = A> {
    type Output;

    fn activate(lhs: A, rhs: B) -> Self::Output;
}

pub trait Activate<Rhs = Self> {
    type Output;

    fn activate(&self, rhs: Rhs) -> Self::Output;

}

pub trait ActivateGradient<Rhs = Self>: Activate<Self::Input> {
    type Input;
    type Delta;

    fn activate_gradient(&self, rhs: &Rhs) -> Self::Delta;
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

