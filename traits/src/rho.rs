/*
    Appellation: rho <module>
    Created At: 2026.01.12:09:50:26
    Contrib: @FL03
*/
use crate::math::Tanh;
use num_traits::{Float, One, Zero};

/// [`Rho`] is a higher-kinded trait that provides a mechanism to apply a function over the
/// elements within a container or structure.
pub trait Rho<T> {
    type Cont<U>;

    fn rho<F, U>(&self, f: F) -> Self::Cont<U>
    where
        F: FnMut(T) -> U;

    fn heavyside(&self) -> Self::Cont<T>
    where
        T: PartialOrd + One + Zero,
    {
        self.rho(|i| if i > T::zero() { T::one() } else { T::zero() })
    }
    fn relu(&self) -> Self::Cont<T>
    where
        T: PartialOrd + Zero,
    {
        self.rho(|i| if i > T::zero() { i } else { T::zero() })
    }

    fn sinh(&self) -> Self::Cont<T>
    where
        T: Float,
    {
        self.rho(|i| i.sinh())
    }

    fn tanh(&self) -> Self::Cont<<T as Tanh>::Output>
    where
        T: Tanh,
    {
        self.rho(|i| i.tanh())
    }
}
/*
 ************* Implementations *************
*/
use ndarray::{Array, ArrayBase, Data, Dimension};

impl<S, D, A> Rho<A> for ArrayBase<S, D, A>
where
    S: Data<Elem = A>,
    D: Dimension,
    A: Clone,
{
    type Cont<U> = Array<U, D>;

    fn rho<F, U>(&self, f: F) -> Self::Cont<U>
    where
        F: FnMut(A) -> U,
    {
        self.mapv(f)
    }
}
