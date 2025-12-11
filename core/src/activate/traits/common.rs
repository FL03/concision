/*
    appellation: unary <module>
    authors: @FL03
*/

macro_rules! unary {
    (@impl $name:ident::$call:ident(self)) => {
        paste::paste! {
            pub trait $name {
                type Output;

                fn $call(self) -> Self::Output;

                fn [<$call _derivative>](self) -> Self::Output;
            }
        }

    };
    (@impl $name:ident::$call:ident(&self)) => {
        paste::paste! {
            pub trait $name {
                type Output;

                fn $call(&self) -> Self::Output;

                fn [<$call _derivative>](&self) -> Self::Output;
            }
        }
    };
    (@impl $name:ident::$call:ident(&mut self)) => {
        paste::paste! {
            pub trait $name {
                type Output;

                fn $call(&mut self) -> Self::Output;

                fn [<$call _derivative>](&mut self) -> Self::Output;
            }
        }
    };
    ($(
        $name:ident::$call:ident($($rest:tt)*)
    ),* $(,)?) => {
        $(
            unary!(@impl $name::$call($($rest)*));
        )*
    };
}

unary! {
    HeavysideActivation::heavyside(self),
    LinearActivation::linear(self),
    SigmoidActivation::sigmoid(self),
    SoftmaxActivation::softmax(&self),
    ReLUActivation::relu(&self),
    TanhActivation::tanh(&self),
}

pub trait SoftmaxAxis: SoftmaxActivation {
    fn softmax_axis(self, axis: usize) -> Self::Output;
}

/*
 ************* Implementations *************
*/

use ndarray::{Array, ArrayBase, Data, Dimension, ScalarOperand};
use num_traits::{Float, One, Zero};

macro_rules! impl_heavyside {
    ($($ty:ty),* $(,)*) => {
        $(
            impl HeavysideActivation for $ty {
                type Output = $ty;

                fn heavyside(self) -> Self::Output {
                    if self > <$ty>::zero() {
                        self
                    } else {
                        <$ty>::zero()
                    }
                }

                fn heavyside_derivative(self) -> Self::Output {
                    if self > <$ty>::zero() {
                        <$ty>::one()
                    } else {
                        <$ty>::zero()
                    }
                }
            }
        )*
    };
}

impl_heavyside!(
    f32, f64, i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize,
);

impl<A, B, S, D> HeavysideActivation for ArrayBase<S, D, A>
where
    A: Clone + HeavysideActivation<Output = B>,
    D: Dimension,
    S: Data<Elem = A>,
{
    type Output = Array<B, D>;

    fn heavyside(self) -> Self::Output {
        self.mapv(HeavysideActivation::heavyside)
    }

    fn heavyside_derivative(self) -> Self::Output {
        self.mapv(HeavysideActivation::heavyside_derivative)
    }
}

impl<A, B, S, D> HeavysideActivation for &ArrayBase<S, D, A>
where
    A: Clone + HeavysideActivation<Output = B>,
    D: Dimension,
    S: Data<Elem = A>,
{
    type Output = Array<B, D>;

    fn heavyside(self) -> Self::Output {
        self.mapv(HeavysideActivation::heavyside)
    }

    fn heavyside_derivative(self) -> Self::Output {
        self.mapv(HeavysideActivation::heavyside_derivative)
    }
}

impl<T> LinearActivation for T
where
    T: Clone + Default,
{
    type Output = T;

    fn linear(self) -> Self::Output {
        self.clone()
    }

    fn linear_derivative(self) -> Self::Output {
        <T>::default()
    }
}

impl<A, S, D> ReLUActivation for ArrayBase<S, D, A>
where
    A: Copy + PartialOrd + Zero + One,
    S: Data<Elem = A>,
    D: Dimension,
{
    type Output = Array<A, D>;

    fn relu(&self) -> Self::Output {
        self.map(|&i| if i > A::zero() { i } else { A::zero() })
    }

    fn relu_derivative(&self) -> Self::Output {
        self.map(|&i| if i > A::zero() { A::one() } else { A::zero() })
    }
}

impl<A, S, D> SigmoidActivation for ArrayBase<S, D, A>
where
    A: 'static + Float,
    S: Data<Elem = A>,
    D: Dimension,
{
    type Output = Array<A, D>;

    fn sigmoid(self) -> Self::Output {
        let dim = self.dim();
        let ones = Array::<A, D>::ones(dim);

        (ones + self.signum().exp()).recip()
    }

    fn sigmoid_derivative(self) -> Self::Output {
        self.mapv(|i| {
            let s = (A::one() + i.neg().exp()).recip();
            s * (A::one() - s)
        })
    }
}

impl<A, S, D> SoftmaxActivation for ArrayBase<S, D, A>
where
    A: ScalarOperand + Float,
    S: Data<Elem = A>,
    D: Dimension,
{
    type Output = Array<A, D>;

    fn softmax(&self) -> Self::Output {
        &self.exp() / self.exp().sum()
    }

    fn softmax_derivative(&self) -> Self::Output {
        let e = self.exp();
        let softmax = &e / e.sum();

        let ones = Array::<A, D>::ones(self.dim());
        &softmax * (&ones - &softmax)
    }
}

impl<A, S, D> TanhActivation for ArrayBase<S, D, A>
where
    A: 'static + Float,
    S: Data<Elem = A>,
    D: Dimension,
{
    type Output = Array<A, D>;

    fn tanh(&self) -> Self::Output {
        self.mapv(|i| i.tanh())
    }

    fn tanh_derivative(&self) -> Self::Output {
        self.mapv(|i| A::one() - i.tanh().powi(2))
    }
}
