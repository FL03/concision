/*
    appellation: hidden <module>
    authors: @FL03
*/
use concision_params::ParamsBase;
use ndarray::{Data, Dimension, RawData};

/// The [`RawHidden`] trait for compatible representations of hidden layers
pub trait RawHidden<S, D>
where
    S: RawData,
    D: Dimension,
{
    private!();

    fn count(&self) -> usize;
}

/// The [`ShallowModelRepr`] trait for shallow neural networks
pub trait ShallowModelRepr<S, D>: RawHidden<S, D>
where
    S: RawData,
    D: Dimension,
{
    private!();
}
/// The [`DeepModelRepr`] trait for deep neural networks
pub trait DeepModelRepr<S, D>: RawHidden<S, D>
where
    S: RawData,
    D: Dimension,
    Self: IntoIterator<Item = ParamsBase<S, D>>,
{
    private!();

    /// returns the hidden layers as a slice
    fn as_slice(&self) -> &[ParamsBase<S, D>];

    /// returns the hidden layers as a mutable slice
    fn as_mut_slice(&mut self) -> &mut [ParamsBase<S, D>];
}

/*
 ************* Implementations *************
*/

impl<X, A, S, D> DeepModelRepr<S, D> for X
where
    S: RawData<Elem = A>,
    D: Dimension,
    X: RawHidden<S, D>
        + IntoIterator<Item = ParamsBase<S, D>>
        + AsRef<[ParamsBase<S, D>]>
        + AsMut<[ParamsBase<S, D>]>,
{
    seal!();

    fn as_slice(&self) -> &[ParamsBase<S, D>] {
        self.as_ref()
    }

    fn as_mut_slice(&mut self) -> &mut [ParamsBase<S, D>] {
        self.as_mut()
    }
}

impl<S, D, T> RawHidden<S, D> for &T
where
    D: Dimension,
    S: RawData,
    T: RawHidden<S, D>,
{
    seal!();

    fn count(&self) -> usize {
        RawHidden::count(*self)
    }
}

impl<S, D, T> RawHidden<S, D> for &mut T
where
    D: Dimension,
    S: RawData,
    T: RawHidden<S, D>,
{
    seal!();

    fn count(&self) -> usize {
        RawHidden::count(*self)
    }
}

impl<A, S, D, const N: usize> RawHidden<S, D> for [ParamsBase<S, D>; N]
where
    D: Dimension,
    S: Data<Elem = A>,
{
    seal!();

    fn count(&self) -> usize {
        N
    }
}

macro_rules! impl_raw_hidden_params {
    (#[count = len] $($rest:tt)*) => {
        impl<S, D> RawHidden<S, D> for $($rest)*
        where
            S: RawData,
            D: Dimension,
        {
            seal!();

            fn count(&self) -> usize {
                self.len()
            }
        }
    };
    (#[count = 1] $($rest:tt)*) => {
        impl<S, D> RawHidden<S, D> for $($rest)*
        where
            S: RawData,
            D: Dimension,
        {
            seal!();

            fn count(&self) -> usize {
                1
            }
        }

        impl<S, D> ShallowModelRepr<S, D> for $($rest)*
        where
            S: RawData,
            D: Dimension,
        {
            seal!();
        }
    };
}

impl_raw_hidden_params! {
    #[count = 1]
    ParamsBase<S, D>
}

impl_raw_hidden_params! {
    #[count = len]
    [ParamsBase<S, D>]
}

impl_raw_hidden_params! {
    #[count = len]
    Vec<ParamsBase<S, D>>
}

impl_raw_hidden_params! {
    #[count = len]
    std::collections::HashSet<ParamsBase<S, D>>
}
