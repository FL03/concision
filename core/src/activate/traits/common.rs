/*
    appellation: unary <module>
    authors: @FL03
*/
use ndarray::{Array, ArrayBase, Data, DataMut, Dimension, ScalarOperand};
use num_traits::{Float, One, Zero};

pub trait SoftmaxAxis: SoftmaxActivation {
    fn softmax_axis(self, axis: usize) -> Self::Output;
}

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

/*
 ************* Implementations *************
*/

macro_rules! impl_heavyside {
    ($($T:ty),* $(,)*) => {
        $(
            impl HeavysideActivation for $T {
                type Output = $T;

                fn heavyside(self) -> Self::Output {
                    if self > <$T>::zero() {
                        <$T>::one()
                    } else {
                        <$T>::zero()
                    }
                }

                fn heavyside_derivative(self) -> Self::Output {
                    if self > <$T>::zero() {
                        <$T>::one()
                    } else {
                        <$T>::zero()
                    }
                }
            }
        )*
    };
}

macro_rules! impl_linear {
    ($($T:ty),* $(,)*) => {
        $(
            impl LinearActivation for $T {
                type Output = $T;

                fn linear(self) -> Self::Output {
                    self
                }

                fn linear_derivative(self) -> Self::Output {
                    <$T>::one()
                }
            }
        )*
    };
}

impl_heavyside!(
    i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize, f32, f64,
);

impl_linear!(
    i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize, f32, f64,
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

impl<A, S, D> LinearActivation for ArrayBase<S, D, A>
where
    A: Clone + One,
    D: Dimension,
    S: DataMut<Elem = A>,
{
    type Output = ArrayBase<S, D, A>;

    fn linear(self) -> Self::Output {
        self
    }

    fn linear_derivative(self) -> Self::Output {
        self.mapv_into(|_| <A>::one())
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
        let exp = self.exp();
        &exp / exp.sum()
    }

    fn softmax_derivative(&self) -> Self::Output {
        let softmax = self.softmax();

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
