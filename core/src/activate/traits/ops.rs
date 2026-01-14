/*
    Appellation: ops <module>
    Created At: 2026.01.13:17:12:10
    Contrib: @FL03
*/
/// Compute the softmax activation along a specified axis.
pub trait SoftmaxAxis: SoftmaxActivation {
    fn softmax_axis(self, axis: usize) -> Self::Output;
}

macro_rules! unary {
    (@impl $name:ident::$call:ident($($rest:tt)*)) => {
        paste::paste! {
            pub trait $name {
                type Output;

                fn $call($($rest)*) -> Self::Output;

                fn [<$call _derivative>]($($rest)*) -> Self::Output;
            }
        }
    };
    ($($name:ident::$call:ident($($rest:tt)*)),* $(,)?) => {
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
