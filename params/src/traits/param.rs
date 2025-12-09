/*
    Appellation: param <module>
    Created At: 2025.12.08:16:03:55
    Contrib: @FL03
*/
/// The [`RawParameter`] trait is used to denote objects capable of being used as a paramater
/// within a neural network or machine learning context. More over, it provides us with an
/// ability to associate some generic element type with the parameter and thus allows us to
/// consider so-called _parameter spaces_. If we allow a parameter space to simply be a
/// collection of points then we can refine the definition downstream to consider specific
/// interpolations, distributions, or manifolds. In other words, we are trying to construct
/// a tangible configuration space for our models so that we can reason about optimization
/// and training in a more formal manner.
///
/// **Note**: This trait is sealed and cannot be implemented outside of this crate.
pub trait RawParameter {
    type Elem: ?Sized;

    private!();
}

/// The [`ScalarParameter`] trait naturally extends the [`RawParameter`] trait to define a
/// scaler as a parameter whose element type is itself. This is useful for defining
/// parameters which are simple scalars such as `f32` or `i64`.
pub trait ScalarParam: RawParameter<Elem = Self> + Sized {
    private!();
}

pub trait TensorParams: RawParameter {
    /// returns the number of dimensions of the parameter
    fn rank(&self) -> usize;
    /// returns the shape of the parameter as a slice
    fn shape(&self) -> &[usize];
    /// returns the size of the parameter
    fn size(&self) -> usize;
}

/*
 ************* Implementations *************
*/
use crate::ParamsBase;
use ndarray::{ArrayBase, Dimension, RawData};

impl<T> RawParameter for &T
where
    T: RawParameter,
{
    type Elem = T::Elem;

    seal! {}
}

impl<T> RawParameter for &mut T
where
    T: RawParameter,
{
    type Elem = T::Elem;

    seal! {}
}

impl<T> ScalarParam for T
where
    T: RawParameter<Elem = T>,
{
    seal!();
}

macro_rules! impl_param {
    ($($T:ty),* $(,)?) => {
        $(impl_param!(@impl $T);)*
    };
    (@impl $T:ty) => {
        impl RawParameter for $T {
            type Elem = $T;

            seal! {}
        }

        impl TensorParams for $T {
            fn rank(&self) -> usize {
                0
            }

            fn shape(&self) -> &[usize] {
                &[]
            }

            fn size(&self) -> usize {
                1
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
impl RawParameter for alloc::string::String {
    type Elem = u8;

    seal! {}
}

impl<S, D, A> RawParameter for ArrayBase<S, D, A>
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

    fn shape(&self) -> &[usize] {
        self.shape()
    }

    fn size(&self) -> usize {
        self.len()
    }
}

impl<S, D, A> RawParameter for ParamsBase<S, D, A>
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

    fn shape(&self) -> &[usize] {
        self.weights().shape()
    }

    fn size(&self) -> usize {
        self.weights().len()
    }
}

