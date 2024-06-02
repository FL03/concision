/*
    Appellation: container <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::traits::ContainerRepr;
use concision::Dimensional;

pub trait Container<T> {
    type Data: ContainerRepr<Elem = T>;
}

/// This trait describes the basic operations for any n-dimensional container.
pub trait NdContainer<A, D>: Dimensional<Dim = D> {
    type Data: ContainerRepr<Elem = A>;

    fn as_slice(&self) -> &[A];

    fn as_mut_slice(&mut self) -> &mut [A];
}

/*
 ************* Implementations *************
*/
impl<S, T> Container<T> for Vec<S> {
    type Data = Vec<T>;
}
