/*
   Appellation: like <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use nd::{ArrayBase, DataOwned, Dimension};
use num::traits::Num;

pub trait ArrayLike<T, O = Self>
where
    Self: DefaultLike<Output = O>
        + FillLike<T, Output = O>
        + OnesLike<Output = O>
        + ZerosLike<Output = O>,
{
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

impl<A, S, D> ArrayLike<A, ArrayBase<S, D>> for ArrayBase<S, D>
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
