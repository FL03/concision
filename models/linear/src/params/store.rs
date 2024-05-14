/*
    Appellation: params <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::{build_bias, Biased, Features, Node, Unbiased};
use core::marker::PhantomData;
use nd::*;
use num::{One, Zero};

///
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
    S: RawData<Elem = A>,
{
    impl_param_builder!(default where A: Default, S: DataOwned);
    impl_param_builder!(ones where A: Clone + One, S: DataOwned);
    impl_param_builder!(zeros where A: Clone + Zero, S: DataOwned);

    pub fn into_biased(self) -> ParamsBase<S, D, Biased>
    where
        A: Default,
        K: 'static,
        S: DataOwned,
    {
        if self.is_biased() {
            return ParamsBase {
                bias: self.bias,
                weights: self.weights,
                _mode: PhantomData::<Biased>,
            };
        }
        let sm = crate::bias_dim(self.raw_dim());
        ParamsBase {
            bias: Some(ArrayBase::default(sm)),
            weights: self.weights,
            _mode: PhantomData::<Biased>,
        }
    }

    pub fn into_unbiased(self) -> ParamsBase<S, D, Unbiased> {
        ParamsBase {
            bias: None,
            weights: self.weights,
            _mode: PhantomData::<Unbiased>,
        }
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
    /// Returns true if the parameter store is biased;
    /// Compares the [TypeId](core::any::TypeId) of the store with the [Biased](crate::Biased) type.
    pub fn is_biased(&self) -> bool
    where
        K: 'static,
    {
        use core::any::TypeId;
        TypeId::of::<Biased>() == TypeId::of::<K>()
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
    ndview!(into_owned::<OwnedRepr>(self) where A: Clone, S: Data);

    ndview!(into_shared::<OwnedArcRepr>(self) where A: Clone, S: DataOwned);

    ndview!(to_owned::<OwnedRepr>(&self) where A: Clone, S: Data);

    ndview!(to_shared::<OwnedArcRepr>(&self) where A: Clone, S: DataOwned);

    ndview!(view::<'a, ViewRepr>(&self) where A: Clone, S: Data);

    ndview!(view_mut::<'a, ViewRepr>(&mut self) where A: Clone, S: DataMut);
}

impl<A, S, D> ParamsBase<S, D, Biased>
where
    D: RemoveAxis,
    S: RawData<Elem = A>,
{
    /// Create a new biased parameter store from the given shape.
    pub fn biased<Sh>(shape: Sh) -> Self
    where
        A: Default,
        S: DataOwned,
        Sh: ShapeBuilder<Dim = D>,
    {
        let dim = shape.into_shape().raw_dim().clone();
        Self {
            bias: build_bias(true, dim.clone(), ArrayBase::default),
            weights: ArrayBase::default(dim),
            _mode: PhantomData::<Biased>,
        }
    }

    pub fn bias(&self) -> &ArrayBase<S, D::Smaller> {
        self.bias.as_ref().unwrap()
    }

    pub fn bias_mut(&mut self) -> &mut ArrayBase<S, D::Smaller> {
        self.bias.as_mut().unwrap()
    }
}

impl<A, S, D> ParamsBase<S, D, Unbiased>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    /// Create a new unbiased parameter store from the given shape.
    pub fn unbiased<Sh>(shape: Sh) -> Self
    where
        A: Default,
        S: DataOwned,
        Sh: ShapeBuilder<Dim = D>,
    {
        Self {
            bias: None,
            weights: ArrayBase::default(shape),
            _mode: PhantomData::<Unbiased>,
        }
    }
}
impl<A, S, K> ParamsBase<S, Ix2, K>
where
    K: 'static,
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

impl<A, S, D> Default for ParamsBase<S, D, Unbiased>
where
    A: Default,
    D: Dimension,
    S: DataOwned<Elem = A>,
{
    fn default() -> Self {
        Self {
            bias: None,
            weights: Default::default(),
            _mode: PhantomData,
        }
    }
}
