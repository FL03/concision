/*
    Appellation: ndtensor <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use nd::{ArrayBase, Data, Dimension, RawData};
use num::complex::ComplexFloat;
use num::traits::Float;

pub trait Scalar {
    type R: Float;
}

pub trait NdTensor<A, D>
where
    A: ComplexFloat,
    D: Dimension,
{
    type Data: RawData<Elem = A>;
    type Output;

    fn conj(&self) -> Self::Output;

    fn cos(&self) -> Self::Output;

    fn cosh(&self) -> Self::Output;
}

/*
 ************* Implementations *************
*/
impl<A, S, D> NdTensor<A, D> for ArrayBase<S, D>
where
    A: ComplexFloat,
    D: Dimension,
    S: Data<Elem = A>,
    Self: Clone,
{
    type Data = S;
    type Output = nd::Array<A, D>;

    fn conj(&self) -> Self::Output {
        self.mapv(|x| x.conj())
    }

    fn cos(&self) -> Self::Output {
        self.mapv(|x| x.cos())
    }

    fn cosh(&self) -> Self::Output {
        self.mapv(|x| x.cosh())
    }
}
