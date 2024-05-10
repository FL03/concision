/*
    Appellation: wnb <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use nd::*;

pub trait WnB<S, D = Ix2>
where
    D: Dimension,
    S: RawData,
{
    fn bias(&self) -> Option<&ArrayBase<S, D::Smaller>>;

    fn bias_mut(&mut self) -> Option<&mut ArrayBase<S, D::Smaller>>;

    fn weight(&self) -> &ArrayBase<S, D>;

    fn weight_mut(&mut self) -> &mut ArrayBase<S, D>;
}

pub trait ParamMode: 'static {
    type Mode;

    fn is_biased(&self) -> bool;

    private!();
}

pub trait IsBiased {
    fn is_biased(&self) -> bool;
}

/*
 ********* Implementations *********
*/

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize,))]
pub enum Biased {}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize,))]
pub enum Unbiased {}

macro_rules! impl_param_ty {
    ($($T:ty: $b:expr),* $(,)?) => {
        $(impl_param_ty!(@impl $T: $b);)*
    };
    (@impl $T:ty: $b:expr) => {
        impl ParamMode for $T {
            type Mode = $T;

            fn is_biased(&self) -> bool {
                $b
            }
            seal!();
        }
    };

}

impl_param_ty!(Biased: true, Unbiased: false,);
