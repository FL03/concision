/*
    Appellation: attention <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::head::AttentionHead;

pub(crate) mod head;

pub mod multi;

pub(crate) mod prelude {
    pub use super::head::AttentionHead;
}
