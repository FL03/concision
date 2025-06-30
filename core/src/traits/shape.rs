/// the [`RawDimension`] trait is used to define a type that can be used as a raw dimension.
/// This trait is primarily used to provide abstracted, generic interpretations of the
/// dimensions of the [`ndarray`] crate to ensure long-term compatibility.
pub trait RawDimension {
    private! {}
}

impl<D> RawDimension for D
where
    D: ndarray::Dimension,
{
    seal! {}
}
