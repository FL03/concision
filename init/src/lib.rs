//! Custom random number distributions and initializers focused on neural networks.
//!
#![crate_name = "concision_init"]
#![crate_type = "lib"]
#![allow(
    clippy::missing_safety_doc,
    clippy::module_inception,
    clippy::needless_doctest_main,
    clippy::should_implement_trait,
    clippy::upper_case_acronyms,
    rustdoc::redundant_explicit_links
)]
#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(all(feature = "alloc", feature = "nightly"), feature(allocator_api))]
// compile-time checks
#[cfg(not(any(feature = "std", feature = "alloc")))]
compiler_error! { "At least one of the \"std\" or \"alloc\" features must be enabled for the crate to compile." }
// external crate
#[cfg(feature = "alloc")]
extern crate alloc;
// re-declarations
#[doc(no_inline)]
pub use rand;
#[doc(no_inline)]
pub use rand_distr;
// modules
pub mod error;

pub mod distr {
    //! random distributions optimized for neural network initialization.
    #[doc(inline)]
    pub use self::{lecun::*, trunc::*, xavier::*};

    mod lecun;
    mod trunc;
    mod xavier;
}

mod traits {
    #[doc(inline)]
    pub use self::ndrand::*;

    mod ndrand;
}

mod utils {
    #[doc(inline)]
    pub use self::rand_utils::*;

    mod rand_utils;
}
// re-exports
#[doc(inline)]
pub use self::{distr::*, error::*, traits::*, utils::*};
// prelude
#[doc(hidden)]
pub mod prelude {
    pub use crate::distr::*;
    pub use crate::traits::*;
    pub use crate::utils::*;
}
