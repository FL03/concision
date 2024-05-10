/*
    Appellation: params <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::{build_bias, Features};
use core::ops;
use nd::linalg::Dot;
use nd::*;
use num::{One, Zero};

#[cfg(feature = "alloc")]
use alloc::vec;
#[cfg(feature = "std")]
use std::vec;

pub(crate) type BuilderResult<S, D = Ix2, E = Ix1> = (ArrayBase<S, D>, Option<ArrayBase<S, E>>);
pub(crate) type Node<A = f64> = (Array<A, Ix1>, Option<Array<A, Ix0>>);

macro_rules! constructor {
    ($call:ident where $($rest:tt)*) => {
        constructor!(@impl $call where $($rest)*);
    };
    (@impl $call:ident where $($rest:tt)*) => {
        pub fn $call<Sh>(biased: bool, shape: Sh) -> ParamsBase<S, D>
        where
            Sh: ndarray::ShapeBuilder<Dim = D>,
            $($rest)*
        {
            let shape = shape.into_shape();
            let dim = shape.raw_dim().clone();
            ParamsBase {
                bias: build_bias(biased, dim.clone(), |dim| ArrayBase::$call(dim)),
                weights: ArrayBase::$call(dim),
            }
        }
    };
}

macro_rules! builder {
    ($call:ident<$($t:ident),* $(,)?> where $($rest:tt)*) => {
        builder!(@impl $call<$($t),*> where $($rest)*);
    };
    (@impl $call:ident<$($t:ident),*> where $($rest:tt)*) => {
        pub fn $call<S, D, Sh, $($t),*>(biased: bool, shape: Sh) -> (ArrayBase<S, D>, Option<ArrayBase<S, D::Smaller>>)
        where
            D: RemoveAxis,
            Sh: ShapeBuilder<Dim = D>,
            $($rest)*
        {
            let shape = shape.into_shape();
            let dim = shape.raw_dim().clone();
            let bias = build_bias(biased, dim.clone(), |dim| ArrayBase::$call(dim));
            let weight = ArrayBase::$call(dim);
            (weight, bias)
        }
    };
}

builder!(default<A> where A: Clone + Default, S: DataOwned<Elem = A>);

pub struct ParamsBase<S, D = Ix2>
where
    D: RemoveAxis,
    S: RawData,
{
    pub(crate) bias: Option<ArrayBase<S, D::Smaller>>,
    pub(crate) weights: ArrayBase<S, D>,
}

impl<A, S, D> ParamsBase<S, D>
where
    D: RemoveAxis,
    S: RawData<Elem = A>,
{
    constructor!(default where A: Default, S: DataOwned);
    constructor!(ones where A: Clone + One, S: DataOwned);
    constructor!(zeros where A: Clone + Zero, S: DataOwned);

    pub fn new<Sh>(shape: Sh) -> Self
    where
        A: Clone + Default,
        S: DataOwned,
        Sh: ShapeBuilder<Dim = D>,
    {
        Self {
            bias: None,
            weights: ArrayBase::default(shape),
        }
    }

    pub fn build<Sh, B>(biased: bool, shape: Sh, builder: B) -> Self
    where
        B: Fn(bool, Sh) -> BuilderResult<S, D, D::Smaller>,
        Sh: ShapeBuilder<Dim = D>,
    {
        let (weights, bias) = builder(biased, shape);
        Self { bias, weights }
    }

    pub fn biased<F>(self, builder: F) -> Self
    where
        F: Fn(D::Smaller) -> ArrayBase<S, D::Smaller>,
    {
        Self {
            bias: build_bias(true, self.raw_dim(), builder),
            ..self
        }
    }

    pub fn unbiased(self) -> Self {
        Self { bias: None, ..self }
    }

    pub fn bias(&self) -> Option<&ArrayBase<S, D::Smaller>> {
        self.bias.as_ref()
    }

    pub fn bias_mut(&mut self) -> Option<&mut ArrayBase<S, D::Smaller>> {
        self.bias.as_mut()
    }

    pub const fn weights(&self) -> &ArrayBase<S, D> {
        &self.weights
    }

    pub fn weights_mut(&mut self) -> &mut ArrayBase<S, D> {
        &mut self.weights
    }

    pub fn features(&self) -> Features {
        Features::from_shape(self.shape())
    }

    pub fn in_features(&self) -> usize {
        self.features().dmodel()
    }

    pub fn is_biased(&self) -> bool {
        self.bias().is_some()
    }

    pub fn linear<T, B>(&self, data: &T) -> B
    where
        A: NdFloat,
        B: for<'a> ops::Add<&'a ArrayBase<S, D::Smaller>, Output = B>,
        S: Data<Elem = A>,
        T: Dot<Array<A, D>, Output = B>,
    {
        let dot = data.dot(&self.weights().t().to_owned());
        if let Some(bias) = self.bias() {
            return dot + bias;
        }
        dot
    }

    pub fn ndim(&self) -> usize {
        self.weights().ndim()
    }

    pub fn out_features(&self) -> usize {
        if self.ndim() == 1 {
            return 1;
        }
        self.shape()[1]
    }
    /// Returns the raw dimension of the weights.
    pub fn raw_dim(&self) -> D {
        self.weights().raw_dim()
    }
    /// Returns the shape of the weights.
    pub fn shape(&self) -> &[usize] {
        self.weights().shape()
    }
}

impl<A, S> ParamsBase<S>
where
    S: RawData<Elem = A>,
{
    pub fn set_node(&mut self, idx: usize, node: Node<A>)
    where
        A: Clone + Default,
        S: DataMut + DataOwned,
    {
        let (weight, bias) = node;
        if let Some(bias) = bias {
            if !self.is_biased() {
                let mut tmp = ArrayBase::default(self.out_features());
                tmp.index_axis_mut(Axis(0), idx).assign(&bias);
                self.bias = Some(tmp);
            }
            self.bias
                .as_mut()
                .unwrap()
                .index_axis_mut(Axis(0), idx)
                .assign(&bias);
        }

        self.weights_mut()
            .index_axis_mut(Axis(0), idx)
            .assign(&weight);
    }
}

impl<A, S> IntoIterator for ParamsBase<S>
where
    A: Clone,
    S: Data<Elem = A>,
{
    type Item = Node<A>;
    type IntoIter = vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        if let Some(bias) = self.bias() {
            return self
                .weights()
                .axis_iter(Axis(0))
                .zip(bias.axis_iter(Axis(0)))
                .map(|(w, b)| (w.to_owned(), Some(b.to_owned())))
                .collect::<Vec<_>>()
                .into_iter();
        }
        self.weights()
            .axis_iter(Axis(0))
            .map(|w| (w.to_owned(), None).into())
            .collect::<Vec<_>>()
            .into_iter()
    }
}

impl<A, S> FromIterator<(Array1<A>, Option<Array0<A>>)> for ParamsBase<S, Ix2>
where
    A: Clone + Default,
    S: DataOwned<Elem = A> + DataMut,
{
    fn from_iter<I: IntoIterator<Item = (Array1<A>, Option<Array0<A>>)>>(nodes: I) -> Self {
        let nodes = nodes.into_iter().collect::<Vec<_>>();
        let mut iter = nodes.iter();
        let node = iter.next().unwrap();
        let shape = Features::new(node.0.shape()[0], nodes.len());
        let mut params = ParamsBase::default(true, shape);
        params.set_node(0, node.clone());
        for (i, node) in iter.into_iter().enumerate() {
            params.set_node(i + 1, node.clone());
        }
        params
    }
}

macro_rules! impl_from {


    (A) => {
        impl<A> From<(Array1<A>, A)> for ParamsBase<OwnedRepr<A>, Ix1>
        where
            A: Clone,
        {
            fn from((weight, bias): (Array1<A>, A)) -> Self {
                let bias = ArrayBase::from_elem((), bias);
                Self {
                    bias: Some(bias),
                    weights: weight,
                }
            }
        }
        impl<A> From<(Array1<A>, Option<A>)> for ParamsBase<OwnedRepr<A>, Ix1>
        where
            A: Clone,
        {
            fn from((weights, bias): (Array1<A>, Option<A>)) -> Self {
                Self {
                    bias: bias.map(|b| ArrayBase::from_elem((), b)),
                    weights,
                }
            }
        }
    };
    ($($bias:ty),*) => {
        $(impl_from!(@impl $bias);)*

    };
    (@impl $b:ty) => {
        impl<A, S, D> From<(ArrayBase<S, D>, Option<$b>)> for ParamsBase<S, D>
        where
            D: RemoveAxis,
            S: RawData<Elem = A>,
        {
            fn from((weights, bias): (ArrayBase<S, D>, Option<$b>)) -> Self {
                Self {
                    bias,
                    weights,
                }
            }
        }

        impl<A, S, D> From<(ArrayBase<S, D>, $b)> for ParamsBase<S, D>
        where
            D: RemoveAxis,
            S: RawData<Elem = A>,
        {
            fn from((weights, bias): (ArrayBase<S, D>, $b)) -> Self {
                Self {
                    bias: Some(bias),
                    weights,
                }
            }
        }
    };
}

impl_from!(A);
impl_from!(ArrayBase<S, D::Smaller>);
