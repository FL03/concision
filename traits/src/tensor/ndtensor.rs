/*
    Appellation: ndtensor <module>
    Created At: 2025.11.26:14:27:51
    Contrib: @FL03
*/

pub trait RawTensorData {
    type Elem;
}

pub trait RawTensor<S, D>
where
    S: RawTensorData<Elem = Self::Elem>,
{
    type Elem;
    type Cont<_R: RawTensorData<Elem = Self::Elem>, _D>: RawTensor<_R, _D, Elem = Self::Elem>;
    /// returns the rank, or _dimensionality_, of the tensor
    fn rank(&self) -> usize;
    /// returns the shape of the tensor
    fn shape(&self) -> &[usize];
    /// returns the total number of elements in the tensor
    fn len(&self) -> usize;

    fn as_ptr(&self) -> *const Self::Elem;

    fn as_mut_ptr(&mut self) -> *mut Self::Elem;
}

pub trait Tensor<S, D>: RawTensor<S, D>
where
    S: RawTensorData<Elem = Self::Elem>,
{
    fn apply<F, U>(&self, f: F) -> Self::Cont<S, D>
    where
        F: Fn(&Self::Elem) -> U;
}

pub trait TensorGrad<S, D>: Tensor<S, D>
where
    S: RawTensorData<Elem = Self::Elem>,
{
    type Delta<_S: RawTensorData<Elem = Self::Elem>, _D>: RawTensor<_S, _D, Elem = Self::Elem>;

    fn grad(&self, rhs: &Self::Delta<S, D>) -> Self::Delta<S, D>;
}
