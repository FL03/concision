/*
    Appellation: container <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::traits::Dimensional;

pub trait Container<T> {
    type Data: Data<Elem = T>;
}

pub trait Data {
    type Elem;
}

/// This trait describes the basic operations for any n-dimensional container.
pub trait NdContainer<A = f64, D = nd::Ix2>: Dimensional<D> {
    type Data: Data<Elem = A>;

    fn as_slice(&self) -> &[A];

    fn as_mut_slice(&mut self) -> &mut [A];
}
