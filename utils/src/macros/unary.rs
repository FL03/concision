macro_rules! unary {
    (@branch $name:ident::$call:ident($($rest:tt)*)) => {
        unary!(@impl $name::$call($($rest)*));
    };
    (@impl $name:ident::$call:ident(self)) => {
        pub trait $name {
            type Output;

            fn $call(self) -> Self::Output;
        }
    };
    (@impl $name:ident::$call:ident(&self)) => {
        pub trait $name {
            type Output;

            fn $call(&self) -> Self::Output;
        }
    };
    (@impl $name:ident::$call:ident(&mut self)) => {
        pub trait $name {
            type Output;

            fn $call(&self) -> Self::Output;
        }
    };
    ($($name:ident::$call:ident($($rest:tt)*)),* $(,)?) => {
        $(
            unary!(@impl $name::$call($($rest)*));
        )*
    };
}
#[allow(unused_macros)]
macro_rules! unary_derivative {
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
        pub trait $name {
            type Output;

            fn $call(&self) -> Self::Output;
        }
    };
    ($($name:ident::$call:ident($($rest:tt)*)),* $(,)?) => {
        $(
            unary!(@impl $name::$call($($rest)*));
        )*
    };
}
