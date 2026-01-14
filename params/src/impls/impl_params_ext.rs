/*
    Appellation: impl_params_ext <module>
    Created At: 2026.01.13:18:32:54
    Contrib: @FL03
*/
use crate::params_base::ParamsBase;

use crate::Params;
use crate::traits::{Biased, Weighted};
use concision_traits::{Apply, FillLike, MapInto, MapTo, OnesLike, ZerosLike};
use core::iter::Once;
use ndarray::{ArrayBase, Data, DataOwned, Dimension, OwnedRepr, RawData};
use num_traits::{One, Zero};
use rspace_traits::RawSpace;

impl<A, S, D> RawSpace for ParamsBase<S, D, A>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    type Elem = A;
}

impl<A, S, D> Weighted<S, D, A> for ParamsBase<S, D, A>
where
    S: RawData<Elem = A>,
    D: Dimension,
{
    type Tensor<_S, _D, _A>
        = ArrayBase<_S, _D, _A>
    where
        _D: Dimension,
        _S: RawData<Elem = _A>;

    fn weights(&self) -> &ArrayBase<S, D, A> {
        self.weights()
    }

    fn weights_mut(&mut self) -> &mut ArrayBase<S, D, A> {
        self.weights_mut()
    }
}

impl<A, S, D> Biased<S, D, A> for ParamsBase<S, D, A>
where
    S: RawData<Elem = A>,
    D: Dimension,
{
    fn bias(&self) -> &ArrayBase<S, D::Smaller, A> {
        self.bias()
    }

    fn bias_mut(&mut self) -> &mut ArrayBase<S, D::Smaller, A> {
        self.bias_mut()
    }
}

impl<S, D> core::ops::Deref for ParamsBase<S, D>
where
    D: Dimension,
    S: RawData,
{
    type Target = ndarray::LayoutRef<S::Elem, D>;

    fn deref(&self) -> &Self::Target {
        self.weights().as_layout_ref()
    }
}

impl<A, S, D> core::fmt::Debug for ParamsBase<S, D, A>
where
    D: Dimension,
    S: Data<Elem = A>,
    A: core::fmt::Debug,
{
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        f.debug_struct("ModelParams")
            .field("bias", self.bias())
            .field("weights", self.weights())
            .finish()
    }
}

impl<A, S, D> core::fmt::Display for ParamsBase<S, D, A>
where
    D: Dimension,
    S: Data<Elem = A>,
    A: core::fmt::Display,
{
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        write!(
            f,
            "{{ bias: {}, weights: {} }}",
            self.bias(),
            self.weights()
        )
    }
}

impl<A, S, D> Clone for ParamsBase<S, D, A>
where
    D: Dimension,
    S: ndarray::RawDataClone<Elem = A>,
    A: Clone,
{
    fn clone(&self) -> Self {
        Self::new(self.bias().clone(), self.weights().clone())
    }
}

impl<A, S, D> Copy for ParamsBase<S, D, A>
where
    D: Dimension + Copy,
    <D as Dimension>::Smaller: Copy,
    S: ndarray::RawDataClone<Elem = A> + Copy,
    A: Copy,
{
}

impl<A, S, D> PartialEq for ParamsBase<S, D, A>
where
    D: Dimension,
    S: Data<Elem = A>,
    A: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.bias() == other.bias() && self.weights() == other.weights()
    }
}

impl<A, S, D> PartialEq<&ParamsBase<S, D, A>> for ParamsBase<S, D, A>
where
    D: Dimension,
    S: Data<Elem = A>,
    A: PartialEq,
{
    fn eq(&self, other: &&ParamsBase<S, D, A>) -> bool {
        self.bias() == other.bias() && self.weights() == other.weights()
    }
}

impl<A, S, D> PartialEq<&mut ParamsBase<S, D, A>> for ParamsBase<S, D, A>
where
    D: Dimension,
    S: Data<Elem = A>,
    A: PartialEq,
{
    fn eq(&self, other: &&mut ParamsBase<S, D, A>) -> bool {
        self.bias() == other.bias() && self.weights() == other.weights()
    }
}

impl<A, S, D> Eq for ParamsBase<S, D, A>
where
    D: Dimension,
    S: Data<Elem = A>,
    A: Eq,
{
}

impl<A, S, D> IntoIterator for ParamsBase<S, D, A>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    type Item = ParamsBase<S, D, A>;
    type IntoIter = Once<ParamsBase<S, D, A>>;

    fn into_iter(self) -> Self::IntoIter {
        core::iter::once(self)
    }
}

impl<A, B, S, D, F> Apply<F> for ParamsBase<S, D, A>
where
    A: Clone,
    D: Dimension,
    S: Data<Elem = A>,
    F: Fn(A) -> B,
{
    type Output = ParamsBase<OwnedRepr<B>, D>;

    fn apply(&self, func: F) -> Self::Output {
        ParamsBase {
            bias: self.bias().mapv(&func),
            weights: self.weights().mapv(&func),
        }
    }
}

impl<A, B, S, D, F> MapInto<F, B> for ParamsBase<S, D, A>
where
    A: Clone,
    D: Dimension,
    S: Data<Elem = A>,
    F: Fn(A) -> B,
{
    type Elem = A;
    type Cont<T> = Params<T, D>;

    fn mapi(self, func: F) -> Self::Cont<B> {
        ParamsBase {
            bias: self.bias().mapv(&func),
            weights: self.weights().mapv(&func),
        }
    }
}

impl<'a, A, B, S, D, F> MapInto<F, B> for &'a ParamsBase<S, D, A>
where
    A: Clone,
    D: Dimension,
    S: Data<Elem = A>,
    F: Fn(A) -> B,
{
    type Elem = A;
    type Cont<T> = Params<T, D>;

    fn mapi(self, func: F) -> Self::Cont<B> {
        ParamsBase {
            bias: self.bias().mapv(&func),
            weights: self.weights().mapv(&func),
        }
    }
}

impl<A, B, S, D, F> MapTo<F, B> for ParamsBase<S, D, A>
where
    A: Clone,
    D: Dimension,
    S: Data<Elem = A>,
    F: Fn(A) -> B,
{
    type Cont<V> = Params<V, D>;
    type Elem = A;

    fn mapt(&self, func: F) -> Self::Cont<B> {
        ParamsBase {
            bias: self.bias().mapv(&func),
            weights: self.weights().mapv(&func),
        }
    }
}

impl<A, S, D> OnesLike for ParamsBase<S, D, A>
where
    D: Dimension,
    S: DataOwned<Elem = A>,
    A: Clone + One,
{
    type Output = ParamsBase<S, D, A>;

    fn ones_like(&self) -> Self::Output {
        ParamsBase {
            bias: self.bias().ones_like(),
            weights: self.weights().ones_like(),
        }
    }
}

impl<A, S, D> ZerosLike for ParamsBase<S, D, A>
where
    D: Dimension,
    S: DataOwned<Elem = A>,
    A: Clone + Zero,
{
    type Output = ParamsBase<S, D, A>;

    fn zeros_like(&self) -> Self::Output {
        ParamsBase {
            bias: self.bias().zeros_like(),
            weights: self.weights().zeros_like(),
        }
    }
}

impl<A, S, D> FillLike<A> for ParamsBase<S, D, A>
where
    D: Dimension,
    S: DataOwned<Elem = A>,
    A: Clone,
{
    type Output = ParamsBase<S, D, A>;

    fn fill_like(&self, elem: A) -> Self::Output {
        ParamsBase {
            bias: self.bias().fill_like(elem.clone()),
            weights: self.weights().fill_like(elem),
        }
    }
}
