/*
    Appellation: nn <module>
    Contrib: @FL03
*/
#[doc(inline)]
pub use self::prelude::*;

pub mod dropout;
pub mod layer;
pub mod stack;

pub mod train {
    #[doc(inline)]
    pub use self::prelude::*;

    pub(crate) mod trainer;

    #[allow(unused)]
    pub(crate) mod prelude {
        pub use super::trainer::*;
    }
}

#[allow(unused_imports)]
pub(crate) mod prelude {
    // pub use super::train::prelude::*;
    pub use super::dropout::*;
    pub use super::layer::*;
    pub use super::stack::*;
}

use crate::traits::{Backward, Forward};

pub trait Model<X>: Backward<X, <Self as Forward<X>>::Output> + Forward<X> {}

pub trait Activate<T> {
    type Output;

    fn activate(&self, input: &T) -> Self::Output;
}

impl<X, Y> Activate<X> for fn(&X) -> Y {
    type Output = Y;

    fn activate(&self, input: &X) -> Self::Output {
        self(input)
    }
}
