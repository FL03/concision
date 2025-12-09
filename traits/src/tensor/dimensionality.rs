/*
    Appellation: shape <module>
    Created At: 2025.11.26:13:10:09
    Contrib: @FL03
*/

/// the [`Dim`] trait is used to define a type that can be used as a raw dimension.
/// This trait is primarily used to provide abstracted, generic interpretations of the
/// dimensions of the [`ndarray`] crate to ensure long-term compatibility.
pub trait RawDimension {
    type Shape;

    private! {}
}

pub trait Dim: RawDimension {
    /// returns the total number of elements considered by the dimension
    fn size(&self) -> usize;
}

/*
 ************* Implementations *************
*/

impl<D> RawDimension for D
where
    D: nd::Dimension,
{
    type Shape = D::Pattern;

    seal! {}
}
