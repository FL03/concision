/*
    Appellation: container <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::traits::Dimensional;

pub trait Container<T> {
    type Data: ContainerData<Elem = T>;
}

pub trait ContainerData {
    type Elem;
}

/// This trait describes the basic operations for any n-dimensional container.
pub trait NdContainer<A = f64, D = nd::Ix2>: Dimensional<D> {
    type Data: ContainerData<Elem = A>;

    fn as_slice(&self) -> &[A];

    fn as_mut_slice(&mut self) -> &mut [A];
}

/*
 ************* Implementations *************
*/
impl<T> ContainerData for Vec<T> {
    type Elem = T;
}

impl<S, T> Container<T> for Vec<S> {
    type Data = Vec<T>;
}
