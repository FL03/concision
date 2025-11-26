/*
    Appellation: shape <module>
    Created At: 2025.11.26:13:10:09
    Contrib: @FL03
*/

/// the [`RawDimension`] trait is used to define a type that can be used as a raw dimension.
/// This trait is primarily used to provide abstracted, generic interpretations of the
/// dimensions of the [`ndarray`] crate to ensure long-term compatibility.
pub trait Dim {
    private! {}
}

impl<D> Dim for D
where
    D: ndarray::Dimension,
{
    seal! {}
}
