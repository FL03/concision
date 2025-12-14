/*
    Appellation: ndtensor <module>
    Created At: 2025.11.26:14:27:51
    Contrib: @FL03
*/
use ndarray::{ArrayBase, Data, DataMut, Dimension, OwnedRepr, RawData, RawDataMut};
use num_traits::Float;

pub trait RawTensor<S, D, A> {
    type Cont<_S, _D, _A>
    where
        _D: Dimension,
        _S: RawData<Elem = _A>;

    fn rank(&self) -> usize;

    fn shape(&self) -> &[usize];

    fn size(&self) -> usize;
}

pub trait NdTensor<S, D, A = <S as RawData>::Elem>: RawTensor<S, D, A>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    fn as_ptr(&self) -> *const A;

    fn as_mut_ptr(&mut self) -> *mut A
    where
        S: RawDataMut;

    fn apply<F, B>(&self, f: F) -> Self::Cont<OwnedRepr<B>, D, B>
    where
        F: FnMut(A) -> B,
        A: Clone,
        S: Data;

    fn powi(&self, n: i32) -> Self::Cont<OwnedRepr<A>, D, A>
    where
        A: Float,
        S: DataMut,
    {
        self.apply(|x| x.powi(n))
    }

    fn exp(&self) -> Self::Cont<OwnedRepr<A>, D, A>
    where
        A: Float,
        S: DataMut,
    {
        self.apply(|x| x.exp())
    }

    fn log(&self) -> Self::Cont<OwnedRepr<A>, D, A>
    where
        A: Float,
        S: DataMut,
    {
        self.apply(|x| x.ln())
    }

    fn ln(&self) -> Self::Cont<OwnedRepr<A>, D, A>
    where
        A: Float,
        S: DataMut,
    {
        self.apply(|x| x.ln())
    }

    fn cos(&self) -> Self::Cont<OwnedRepr<A>, D, A>
    where
        A: Float,
        S: DataMut,
    {
        self.apply(|x| x.cos())
    }

    fn cosh(&self) -> Self::Cont<OwnedRepr<A>, D, A>
    where
        A: Float,
        S: DataMut,
    {
        self.apply(|x| x.cosh())
    }

    fn sin(&self) -> Self::Cont<OwnedRepr<A>, D, A>
    where
        A: Float,
        S: DataMut,
    {
        self.apply(|x| x.sin())
    }

    fn sinh(&self) -> Self::Cont<OwnedRepr<A>, D, A>
    where
        A: Float,
        S: DataMut,
    {
        self.apply(|x| x.sinh())
    }

    fn tan(&self) -> Self::Cont<OwnedRepr<A>, D, A>
    where
        A: Float,
        S: DataMut,
    {
        self.apply(|x| x.tan())
    }

    fn tanh(&self) -> Self::Cont<OwnedRepr<A>, D, A>
    where
        A: Float,
        S: DataMut,
    {
        self.apply(|x| x.tanh())
    }
}

pub trait NdGradient<S, D, A = <S as RawData>::Elem>: NdTensor<S, D, A>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    type Delta<_S, _D, _A>: NdTensor<_S, _D, _A>
    where
        _D: Dimension,
        _S: RawData<Elem = _A>;

    fn grad(&self, rhs: &Self::Delta<S, D, A>) -> Self::Delta<S, D, A>;
}

/*
 ************* Implementations *************
*/

impl<A, S, D> RawTensor<S, D, A> for ArrayBase<S, D, A>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    type Cont<_S, _D, _A>
        = ArrayBase<_S, _D, _A>
    where
        _D: Dimension,
        _S: RawData<Elem = _A>;

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

impl<A, S, D> NdTensor<S, D, A> for ArrayBase<S, D, A>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    fn as_ptr(&self) -> *const A {
        self.as_ptr()
    }

    fn as_mut_ptr(&mut self) -> *mut A
    where
        S: RawDataMut,
    {
        self.as_mut_ptr()
    }

    fn apply<F, B>(&self, f: F) -> Self::Cont<OwnedRepr<B>, D, B>
    where
        A: Clone,
        F: FnMut(A) -> B,
        S: Data,
    {
        self.mapv(f)
    }
}
