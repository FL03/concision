/*
    Appellation: ndarray <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use concision::Dimensional;
use nd::{ArrayBase, DataOwned, Dimension, RawData, ShapeBuilder};
use num::{One, Zero};

/// [NdBuilder] describes common creation routines for [ArrayBase]
pub trait NdBuilder<A = f64, D = nd::Ix2>
where
    D: Dimension,
{
    type Data: RawData<Elem = A>;

    /// Create a new array with the given shape whose elements are set to the default value of the element type.
    fn default<Sh>(shape: Sh) -> Self
    where
        A: Default,
        Sh: ShapeBuilder<Dim = D>,
        Self::Data: DataOwned;

    fn fill<Sh>(shape: Sh, elem: A) -> Self
    where
        A: Clone,
        Sh: ShapeBuilder<Dim = D>,
        Self::Data: DataOwned;

    fn ones<Sh>(shape: Sh) -> Self
    where
        A: Clone + One,
        Sh: ShapeBuilder<Dim = D>,
        Self::Data: DataOwned;

    fn zeros<Sh>(shape: Sh) -> Self
    where
        A: Clone + Zero,
        Sh: ShapeBuilder<Dim = D>,
        Self::Data: DataOwned;
}

pub trait NdBuilderExt<A = f64, D = nd::Ix2>: NdBuilder<A, D> + Sized
where
    D: Dimension,
{
    fn dim(&self) -> D::Pattern;

    fn default_like<Sh>(&self) -> Self
    where
        A: Default,
        Sh: ShapeBuilder<Dim = D>,
        Self::Data: DataOwned,
    {
        Self::default(self.dim())
    }

    fn fill_like<Sh>(&self, elem: A) -> Self
    where
        A: Clone,
        Sh: ShapeBuilder<Dim = D>,
        Self::Data: DataOwned,
    {
        Self::fill(self.dim(), elem)
    }

    fn ones_like<Sh>(&self) -> Self
    where
        A: Clone + One,
        Sh: ShapeBuilder<Dim = D>,
        Self::Data: DataOwned,
    {
        Self::ones(self.dim())
    }

    fn zeros_like<Sh>(&self) -> Self
    where
        A: Clone + Zero,
        Sh: ShapeBuilder<Dim = D>,
        Self::Data: DataOwned,
    {
        Self::zeros(self.dim())
    }
}

/*
 ************* Implementations *************
*/
impl<A, S, D> NdBuilder<A, D> for ArrayBase<S, D>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    type Data = S;

    fn default<Sh>(shape: Sh) -> Self
    where
        A: Default,
        Sh: ShapeBuilder<Dim = D>,
        Self::Data: DataOwned,
    {
        ArrayBase::default(shape)
    }

    fn fill<Sh>(shape: Sh, elem: A) -> Self
    where
        A: Clone,
        S: DataOwned,
        Sh: ShapeBuilder<Dim = D>,
    {
        ArrayBase::from_elem(shape, elem)
    }

    fn ones<Sh>(shape: Sh) -> Self
    where
        A: Clone + One,
        Sh: ShapeBuilder<Dim = D>,
        Self::Data: DataOwned,
    {
        ArrayBase::ones(shape)
    }

    fn zeros<Sh>(shape: Sh) -> Self
    where
        A: Clone + Zero,
        Sh: ShapeBuilder<Dim = D>,
        Self::Data: DataOwned,
    {
        ArrayBase::zeros(shape)
    }
}

impl<U, A, D> NdBuilderExt<A, D> for U
where
    U: Dimensional<Dim = D> + NdBuilder<A, D>,
    D: Dimension,
{
    fn dim(&self) -> D::Pattern {
        self.dim()
    }
}
