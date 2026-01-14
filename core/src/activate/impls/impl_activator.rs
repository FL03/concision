/*
    appellation: activate <module>
    authors: @FL03
*/
use crate::activate::Activator;

impl<X, Y, F> Activator<X> for F
where
    F: Fn(X) -> Y,
{
    type Output = Y;

    fn activate(&self, rhs: X) -> Self::Output {
        self(rhs)
    }
}

// impl<F, S, D, A, B> Activator<ArrayBase<S, D, A>> for F
// where
//     F: Activator<A, Output = B>,
//     S: Data<Elem = A>,
//     D: Dimension,
// {
//     type Output = Array<B, D>;

//     fn activate(&self, rhs: ArrayBase<S, D, A>) -> Self::Output {
//         rhs.mapv(|x| self.activate(x))
//     }
// }

#[cfg(feature = "alloc")]
impl<X, Y> Activator<X> for alloc::boxed::Box<dyn Activator<X, Output = Y>> {
    type Output = Y;

    fn activate(&self, rhs: X) -> Self::Output {
        self.as_ref().activate(rhs)
    }
}
