/*
   Appellation: matmul <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use ndarray::linalg::Dot;

pub trait Matmul<Rhs = Self> {
    type Output;

    fn matmul(&self, rhs: &Rhs) -> Self::Output;
}

impl<A, B, C> Matmul<B> for A
where
    A: Dot<B, Output = C>,
{
    type Output = C;

    fn matmul(&self, rhs: &B) -> Self::Output {
        self.dot(rhs)
    }
}
