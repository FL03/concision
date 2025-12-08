/*
    Appellation: param <module>
    Created At: 2025.12.08:16:03:55
    Contrib: @FL03
*/

/// The [`RawParameter`] trait is used to denote objects capable of being used as a paramater
/// within a neural network or machine learning context. More over, it provides us with an
/// ability to associate some generic element type with the parameter and thus allows us to
/// consider so-called _parameter spaces_. If we allow a parameter space to simply be a
/// collection of points then we can refine the definition downstream to consider specific
/// interpolations, distributions, or manifolds. In other words, we are trying to construct
/// a tangible configuration space for our models so that we can reason about optimization
/// and training in a more formal manner.
///
/// **Note**: This trait is sealed and cannot be implemented outside of this crate.
pub trait RawParameter {
    type Elem;

    private!();
}

/// The [`ScalarParameter`] trait naturally extends the [`RawParameter`] trait to define a
/// scaler as a parameter whose element type is itself. This is useful for defining
/// parameters which are simple scalars such as `f32` or `i64`.
pub trait ScalarParameter: RawParameter<Elem = Self> + Sized {
    private!();
}

/*
 ************* Implementations *************
*/

impl<T> ScalarParameter for T
where
    T: RawParameter<Elem = T>,
{
    seal!();
}

macro_rules! impl_param {
    ($($T:ty),* $(,)?) => {
        $(impl_param!(@impl $T);)*
    };
    (@impl $T:ty) => {
        impl RawParameter for $T {
            type Elem = $T;

            seal! {}
        }
    };
}

impl_param! {
    u8, u16, u32, u64, u128, usize,
    i8, i16, i32, i64, i128, isize,
    f32, f64,
}
