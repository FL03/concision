/*
    Appellation: tensor <module>
    Created At: 2025.12.08:16:03:55
    Contrib: @FL03
*/

/// The [`RawTensor`] trait is used to denote objects capable of being used as a paramater
/// within a neural network or machine learning context. More over, it provides us with an
/// ability to associate some generic element type with the parameter and thus allows us to
/// consider so-called _parameter spaces_. If we allow a parameter space to simply be a
/// collection of points then we can refine the definition downstream to consider specific
/// interpolations, distributions, or manifolds. In other words, we are trying to construct
/// a tangible configuration space for our models so that we can reason about optimization
/// and training in a more formal manner.
///
/// **Note**: This trait is sealed and cannot be implemented outside of this crate.
pub trait RawTensor {
    type Elem: ?Sized;

    private! {}
}

/// The [`ScalarTensor`] is a marker trait automatically implemented for
pub trait ScalarTensor: RawTensor<Elem = Self> + Sized {
    private!();
}

pub trait TensorParams: RawTensor {
    /// returns the number of dimensions of the parameter
    fn rank(&self) -> usize;
    /// returns the size of the parameter
    fn size(&self) -> usize;
}

pub trait ExactDimTensor: TensorParams {
    type Shape: ?Sized;
    /// returns the shape of the parameter as a slice
    fn shape(&self) -> &Self::Shape;
}

pub trait DimConst<const N: usize> {}

/*
 ************* Implementations *************
*/
use crate::ParamsBase;
use ndarray::{ArrayBase, Dimension, RawData};

impl<A, T> RawTensor for &T
where
    T: RawTensor<Elem = A>,
{
    type Elem = A;

    seal! {}
}

impl<A, T> RawTensor for &mut T
where
    T: RawTensor<Elem = A>,
{
    type Elem = A;

    seal! {}
}

impl<T> ScalarTensor for T
where
    T: RawTensor<Elem = T>,
{
    seal! {}
}

macro_rules! impl_param {
    ($($T:ty),* $(,)?) => {
        $(impl_param!(@impl $T);)*
    };
    (@impl $T:ty) => {
        impl RawTensor for $T {
            type Elem = $T;

            seal! {}
        }

        impl TensorParams for $T {
            fn rank(&self) -> usize {
                0
            }

            fn size(&self) -> usize {
                1
            }
        }

        impl ExactDimTensor for $T {
            type Shape = [usize; 0];

            fn shape(&self) -> &Self::Shape {
                &[]
            }
        }
    };
}

impl_param! {
    u8, u16, u32, u64, u128, usize,
    i8, i16, i32, i64, i128, isize,
    f32, f64,
    bool, char, str
}

#[cfg(feature = "alloc")]
impl RawTensor for alloc::string::String {
    type Elem = u8;

    seal! {}
}

impl<S, D, A> RawTensor for ArrayBase<S, D, A>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    type Elem = A;

    seal! {}
}

impl<S, D, A> TensorParams for ArrayBase<S, D, A>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    fn rank(&self) -> usize {
        self.ndim()
    }

    fn size(&self) -> usize {
        self.len()
    }
}

impl<S, D, A> ExactDimTensor for ArrayBase<S, D, A>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    type Shape = [usize];

    fn shape(&self) -> &[usize] {
        self.shape()
    }
}

impl<S, D, A> RawTensor for ParamsBase<S, D, A>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    type Elem = A;

    seal! {}
}

impl<S, D, A> TensorParams for ParamsBase<S, D, A>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    fn rank(&self) -> usize {
        self.weights().ndim()
    }

    fn size(&self) -> usize {
        self.weights().len()
    }
}

impl<S, D, A> ExactDimTensor for ParamsBase<S, D, A>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    type Shape = [usize];

    fn shape(&self) -> &[usize] {
        self.weights().shape()
    }
}

impl<T> RawTensor for [T] {
    type Elem = T;

    seal! {}
}

impl<T> RawTensor for &[T] {
    type Elem = T;

    seal! {}
}

impl<T> RawTensor for &mut [T] {
    type Elem = T;

    seal! {}
}

impl<const N: usize, T> RawTensor for [T; N] {
    type Elem = T;

    seal! {}
}

impl<const N: usize, T> TensorParams for [T; N] {
    fn rank(&self) -> usize {
        1
    }

    fn size(&self) -> usize {
        N
    }
}

impl<const N: usize, T> ExactDimTensor for [T; N] {
    type Shape = [usize; 1];

    fn shape(&self) -> &Self::Shape {
        &[N]
    }
}

#[cfg(feature = "alloc")]
mod impl_alloc {
    use super::*;
    use alloc::vec::Vec;

    impl<T> RawTensor for Vec<T>
    where
        T: RawTensor,
    {
        type Elem = T::Elem;

        seal! {}
    }
}
