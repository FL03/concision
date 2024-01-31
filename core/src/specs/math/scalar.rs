/*
   Appellation: scalar <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
// use super::Algebraic;
use num::traits::{NumAssignOps, NumOps};

pub trait Scalar {
    type Complex: NumAssignOps + NumOps + NumOps<Self::Real>;
    type Real: NumAssignOps + NumOps + NumOps<Self::Complex, Self::Complex>;
}
