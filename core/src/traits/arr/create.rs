/*
   Appellation: create <module>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use nd::{ArrayBase, DataOwned, Dimension, Ix2, ShapeBuilder};
use num::traits::Num;

pub trait NdLike<T, O = Self>
where
    Self: DefaultLike<Output = O>
        + FillLike<T, Output = O>
        + OnesLike<Output = O>
        + ZerosLike<Output = O>,
{
}

pub trait ArrayLike<A = f64, D = Ix2>
where
    D: Dimension,
{
    type Output;

    fn array_like<Sh>(&self, shape: Sh, elem: A) -> Self::Output
    where
        Sh: ShapeBuilder<Dim = D>;
}

macro_rules! ndlike {
    ($($name:ident::$(<$($T:ident),*>::)?$method:ident $(($($field:ident:$ft:ty),*))?),* $(,)?) => {
        $(ndlike!(@impl $name::$(<$($T),*>::)?$method$(($($field:$ft),*))?);)*
    };
    (@impl $name:ident::$(<$($T:ident),*>::)?$method:ident$(($($field:ident: $ft:ty),*))?) => {
        pub trait $name$(<$($T),*>)? {
            type Output;

            fn $method(&self $(, $($field:$ft),*)?) -> Self::Output;
        }
    };

}

ndlike!(DefaultLike::default_like, OnesLike::ones_like, ZerosLike::zeros_like, FillLike::<T>::fill_like(elem: T));

/*
 ******** implementations ********
*/

impl<A, S, D> NdLike<A, ArrayBase<S, D>> for ArrayBase<S, D>
where
    A: Clone + Default + Num,
    D: Dimension,
    S: DataOwned<Elem = A>,
{
}

impl<A, S, D> ArrayLike<A, D> for ArrayBase<S, D>
where
    A: Clone,
    D: Dimension,
    S: nd::DataOwned<Elem = A>,
{
    type Output = ArrayBase<S, D>;

    fn array_like<Sh>(&self, shape: Sh, elem: A) -> Self::Output
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

macro_rules! impl_ndlike {

    ($name:ident::$method:ident.$call:ident: $($p:tt)*) => {
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

impl_ndlike!(DefaultLike::default_like.default: Default);
impl_ndlike!(OnesLike::ones_like.ones: Clone + num::One);
impl_ndlike!(ZerosLike::zeros_like.zeros: Clone + num::Zero);
