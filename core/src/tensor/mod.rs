/*
    appellation: tensor <module>
    authors: @FL03
*/
//! this module focuses on establishing a solid foundation for working with n-dimensional
//! tensors.

#[doc(inline)]
pub use self::{tensor::*, traits::*, types::*};

/// this module defines various iterators for the [`TensorBase`]
pub mod iter;

mod tensor;

mod impls {
    mod impl_tensor;
    mod impl_tensor_iter;
    mod impl_tensor_ops;
    mod impl_tensor_repr;

    #[allow(deprecated)]
    mod impl_tensor_deprecated;
    #[cfg(feature = "init")]
    mod impl_tensor_init;
    #[cfg(feature = "rand")]
    mod impl_tensor_rand;
    #[cfg(feature = "serde")]
    mod impl_tensor_serde;
}

mod traits {
    //! this module provides additional traits for the `tensor` module
    #[doc(inline)]
    pub use self::prelude::*;

    mod raw_tensor;
    mod scalar;

    mod prelude {
        #[doc(inline)]
        pub use super::raw_tensor::*;
        #[doc(inline)]
        pub use super::scalar::*;
    }
}

mod types {
    //! this module defines various type aliases and primitives used by the `tensor` module
    #[doc(inline)]
    pub use self::prelude::*;

    mod aliases;

    mod prelude {
        #[doc(inline)]
        pub use super::aliases::*;
    }
}

pub(crate) mod prelude {
    #[doc(inline)]
    pub use super::tensor::*;
    #[doc(inline)]
    pub use super::traits::*;
    #[doc(inline)]
    pub use super::types::*;
}
