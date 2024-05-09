/*
   Appellation: like <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use nd::{ArrayBase, DataOwned, Dimension, RawData, ShapeBuilder};
use num::traits::Num;

pub trait TensorConstructor<T, O = Self>
where
    Self: DefaultLike<Output = O>
        + FillLike<T, Output = O>
        + OnesLike<Output = O>
        + ZerosLike<Output = O>,
{
}

pub trait ArrayLike<A, S, D>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    fn array_like<Sh>(&self, shape: Sh, elem: A) -> ArrayBase<S, D>
    where
        Sh: ShapeBuilder<Dim = D>;
}

impl<A, S, D> ArrayLike<A, S, D> for ArrayBase<S, D>
where
    A: Clone,
    D: Dimension,
    S: nd::DataOwned<Elem = A>,
{
    fn array_like<Sh>(&self, shape: Sh, elem: A) -> ArrayBase<S, D>
    where
        Sh: ShapeBuilder<Dim = D>,
    {
        if self.is_standard_layout() {
            ArrayBase::from_elem(shape, elem)
        } else {
            ArrayBase::from_elem(shape.f(), elem)
        }
    }
}

pub trait DefaultLike {
    type Output;

    fn default_like(&self) -> Self::Output;
}

pub trait FillLike<T> {
    type Output;

    fn fill_like(&self, elem: T) -> Self::Output;
}

pub trait OnesLike {
    type Output;

    fn ones_like(&self) -> Self::Output;
}

pub trait ZerosLike {
    type Output;

    fn zeros_like(&self) -> Self::Output;
}

/*
 ******** implementations ********
*/

impl<A, S, D> TensorConstructor<A, ArrayBase<S, D>> for ArrayBase<S, D>
where
    A: Clone + Default + Num,
    D: Dimension,
    S: DataOwned<Elem = A>,
{
}

impl<A, S, D> FillLike<A> for ArrayBase<S, D>
where
    A: Clone,
    D: Dimension,
    S: DataOwned<Elem = A>,
{
    type Output = ArrayBase<S, D>;

    fn fill_like(&self, elem: A) -> Self::Output {
        ArrayBase::from_elem(self.dim(), elem)
    }
}

macro_rules! impl_like {
    ($name:ident::$method:ident.$call:ident: $($p:tt)*) => {
        impl_like!(@impl $name::$method.$call: $($p)*);
    };
    (@impl $name:ident::$method:ident.$call:ident: $($p:tt)*) => {
        impl<A, S, D> $name for ArrayBase<S, D>
        where
            A: $($p)*,
            D: Dimension,
            S: DataOwned<Elem = A>,
        {
            type Output = ArrayBase<S, D>;

            fn $method(&self) -> Self::Output {
                ArrayBase::$call(self.dim())
            }
        }
    };
}

impl_like!(DefaultLike::default_like.default: Default);
impl_like!(OnesLike::ones_like.ones: Clone + num::One);
impl_like!(ZerosLike::zeros_like.zeros: Clone + num::Zero);
