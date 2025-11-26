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
    Heavyside::heavyside(self),
    LinearActivation::linear(self),
    Sigmoid::sigmoid(self),
    Softmax::softmax(&self),
    ReLU::relu(&self),
    TanhActivation::tanh(&self),
}

pub trait SoftmaxAxis: Softmax {
    fn softmax_axis(self, axis: usize) -> Self::Output;
}
