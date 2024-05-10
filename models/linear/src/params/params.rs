/*
    Appellation: params <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::{build_bias, Features};
use crate::params::mode::*;
use core::marker::PhantomData;
use nd::*;
use num::{One, Zero};

pub(crate) type Pair<A, B = A> = (A, B);
pub(crate) type MaybePair<A, B = A> = Pair<A, Option<B>>;
pub(crate) type Node<A = f64, D = Ix1, E = Ix0> = MaybePair<Array<A, D>, Array<A, E>>;

pub struct ParamsBase<S = OwnedRepr<f64>, D = Ix2, K = Unbiased>
where
    D: Dimension,
    S: RawData,
{
    pub(crate) bias: Option<ArrayBase<S, D::Smaller>>,
    pub(crate) weights: ArrayBase<S, D>,
    pub(crate) _mode: PhantomData<K>,
}

impl<A, S, D, K> ParamsBase<S, D, K>
where
    D: RemoveAxis,
    K: ParamMode,
    S: RawData<Elem = A>,
{
    impl_param_builder!(default where A: Default, S: DataOwned);
    impl_param_builder!(ones where A: Clone + One, S: DataOwned);
    impl_param_builder!(zeros where A: Clone + Zero, S: DataOwned);

    pub fn new<Sh>(shape: Sh) -> Self
    where
        A: Default,
        S: DataOwned,
        Sh: ShapeBuilder<Dim = D>,
    {
        Self {
            bias: None,
            weights: ArrayBase::default(shape),
            _mode: PhantomData,
        }
    }

    pub fn build<F, Sh>(shape: Sh, builder: F) -> Self
    where
        F: Fn(Sh) -> ArrayBase<S, D>,
        Sh: ShapeBuilder<Dim = D>,
        
    {
        let _weights = builder(shape);
        unimplemented!()
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

impl<A, S, D> Default for ParamsBase<S, D> where A: Default, D: Dimension, S: DataOwned<Elem = A> {
    fn default() -> Self {
        Self {
            bias: None,
            weights: Default::default(),
            _mode: PhantomData,
        }
    }
}


