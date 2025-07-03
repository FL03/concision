/*
    appellation: layout <module>
    authors: @FL03
*/
#[doc(inline)]
pub use self::{features::ModelFeatures, format::ModelFormat, traits::*};

mod features;
mod format;

mod traits {
    #[doc(inline)]
    pub use self::prelude::*;

    mod layout;

    mod prelude {
        #[doc(inline)]
        pub use super::layout::*;
    }
}

pub(crate) mod prelude {
    #[doc(inline)]
    pub use super::features::*;
    #[doc(inline)]
    pub use super::format::*;
    #[doc(inline)]
    pub use super::traits::*;
}
