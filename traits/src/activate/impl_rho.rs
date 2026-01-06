/*
    Appellation: impl_rho <module>
    Created At: 2025.12.26:23:59:48
    Contrib: @FL03
*/
// use super::Rho;
// use crate::Apply;

// impl<A, B, C, F> Rho<A> for C
// where
//     C: Apply<F, B, Elem = A>,
//     F: Fn(A) -> B,
// {
//     type Cont<_V> = <C>::Cont<_V>;

//     fn rho(&self, f: F) -> Self::Cont<B>    {
//         self.apply(|x| f(x))
//     }
// }

// #[cfg(feature = "complex")]
// impl<U, S> crate::activate::RhoComplex<U> for S
// where
//     S: Apply<U>,
//     U: num_complex::ComplexFloat,
// {
// }
