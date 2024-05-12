/*
    Appellation: params <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::{item::*, store::ParamsBase};

pub(crate) mod item;
pub(crate) mod store;

macro_rules! params_ty {
    ($target:ident: [$($name:ident<$(&$lt:lifetime)?$repr:ident>),* $(,)?]) => {
        $(params_ty!(@impl $target: $name<$(&$lt)? $repr>);)*
    };
    (@impl $target:ident: $name:ident<$repr:ident>) => {
        pub type $name<A = f64, D = ndarray::Ix2> = $target<ndarray::$repr<A>, D>;
    };
    (@impl $target:ident: $name:ident<&'a $repr:ident>) => {
        pub type $name<'a, A = f64, D = ndarray::Ix2> = $target<ndarray::$repr<&'a A>, D>;
    };
}

params_ty!(
    ParamsBase: [
        Params<OwnedRepr>,
        ArcParams<OwnedArcRepr>,
        ParamsView<&'a ViewRepr>,
    ]
);

#[allow(unused_imports)]
pub(crate) mod prelude {
    pub use super::item::{Entry, QKV};
    pub use super::store::ParamsBase;
    pub use super::{ArcParams, Params};
}
