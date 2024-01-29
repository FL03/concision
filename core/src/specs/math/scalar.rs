/*
   Appellation: scalar <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
// use super::Algebraic;
use num::traits::NumOps;

pub trait Scalar {
    type Complex: NumOps + NumOps<Self::Real>;
    type Real: NumOps + NumOps<Self::Complex, Self::Complex>;
}
