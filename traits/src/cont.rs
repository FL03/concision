/*
    Appellation: store <module>
    Created At: 2025.12.10:21:30:34
    Contrib: @FL03
*/
#[doc(inline)]
pub use self::containers::*;

mod containers;

/// The [`RawStore`] trait provides a generalized interface for all _containers_. The trait is
/// sealed, preventing any external implementations and is primarily used as the basis for
/// other traits, such as [`Sequential`].
pub trait RawStore {
    type Elem;

    private!();
}
/// The [`Sequential`] trait is a marker trait defining a sequential collection of elements.
/// It is sealed, preventing external implementations, and is used to indicate that a type can
/// be treated as a sequence of elements, such as arrays or vectors.
pub trait Sequential {
    private!();
}
/*
    ************* Implementations *************
*/

