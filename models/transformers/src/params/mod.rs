/*
    Appellation: params <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::qkv::QKVBase;

mod qkv;

macro_rules! params_ty {
    ($target:ident: [$($name:ident<$repr:ident>),* $(,)?]) => {
        $(params_ty!(@impl $target, $name, $repr);)*
    };
    (@impl $target:ident, $name:ident, $repr:ident) => {
        pub type $name<A = f64, D = ndarray::Ix2> = $target<ndarray::$repr<A>, D>;
    };
}

params_ty!(
    QKVBase: [
        QKV<OwnedRepr>,
        ArcQKV<OwnedArcRepr>,
    ]
);

pub(crate) mod prelude {
    pub use super::{ArcQKV, QKV};
    pub use super::QKVBase;
}
