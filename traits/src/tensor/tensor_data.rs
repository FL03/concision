/*
    Appellation: repr <module>
    Created At: 2025.12.09:10:04:11
    Contrib: @FL03
*/
use rspace_traits::RawSpace;

pub trait RawTensorData: RawSpace {
    private! {}
}

pub trait RawTensor<S, D, A>
where
    S: RawTensorData,
{
    type Cont<_S, _D, _A>
    where
        _D: ?Sized,
        _S: RawTensorData<Elem = _A>;
}
/// A marker trait used to denote tensors that represent scalar values; more specifically, we
/// consider _**any**_ type implementing the [`RawTensorData`] type where the `Elem` associated
/// type is the implementor itself a scalar value.
pub trait ScalarTensorData: RawTensorData<Elem = Self> {
    private! {}
}

/*
 ************* Implementations *************
*/

impl<T> ScalarTensorData for T
where
    T: RawTensorData<Elem = Self>,
{
    seal! {}
}

macro_rules! impl_scalar_tensor {
    {$($T:ty),* $(,)?} => {
        $(
            impl RawTensorData for $T {
                seal! {}
            }
        )*
    };
}

impl_scalar_tensor! {
    u8, u16, u32, u64, u128, usize,
    i8, i16, i32, i64, i128, isize,
    f32, f64,
    bool, char
}

#[cfg(feature = "alloc")]
impl RawTensorData for alloc::string::String {
    seal! {}
}
