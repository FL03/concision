/*
    Appellation: impl_rho <module>
    Created At: 2025.12.26:23:59:48
    Contrib: @FL03
*/
use super::Rho;
use crate::Apply;

impl<T, S> Rho<T> for S
where
    S: Apply<T>,
{
    type Cont<_V> = S::Cont<_V>;

    fn rho<F, V>(&self, f: F) -> Self::Cont<V>
    where
        F: Fn(T) -> V,
    {
        self.apply(|x| f(x))
    }
}

#[cfg(feature = "complex")]
impl<U, S> crate::activate::RhoComplex<U> for S
where
    S: Apply<U>,
    U: num_complex::ComplexFloat,
{
}
