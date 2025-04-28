#[doc(inline)]
pub use self::dataloader::Dataloader;

pub mod dataloader;

pub(crate) mod prelude {
    #[doc(inline)]
    pub use super::dataloader::*;
}

pub trait Compile {}

pub trait Loader {}
