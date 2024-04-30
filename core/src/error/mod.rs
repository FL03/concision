/*
   Appellation: error <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::prelude::*;

mod err;

pub mod kinds {
    pub use self::prelude::*;

    pub mod external;

    pub(crate) mod prelude {
        pub use super::external::*;
    }
}

pub(crate) mod prelude {
    pub use super::err::*;
    pub use super::kinds::prelude::*;
}
