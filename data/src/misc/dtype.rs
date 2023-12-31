/*
   Appellation: dtype <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/

pub trait DataType {
    fn dtype(&self) -> DType;
}

impl<T> DataType for T
where
    T: Clone + Into<DType>,
{
    fn dtype(&self) -> DType {
        self.clone().into()
    }
}

pub enum DType {
    FloatingPoint(FloatingPoint),
    Integer(Integer),
    Unsigned(Unsigned),
}

impl From<f32> for DType {
    fn from(_: f32) -> Self {
        DType::FloatingPoint(FloatingPoint::F32)
    }
}

impl From<f64> for DType {
    fn from(_: f64) -> Self {
        DType::FloatingPoint(FloatingPoint::F64)
    }
}

pub enum FloatingPoint {
    F32,
    F64,
}

impl From<f32> for FloatingPoint {
    fn from(_: f32) -> Self {
        FloatingPoint::F32
    }
}

impl From<f64> for FloatingPoint {
    fn from(_: f64) -> Self {
        FloatingPoint::F64
    }
}

impl From<FloatingPoint> for DType {
    fn from(dtype: FloatingPoint) -> Self {
        DType::FloatingPoint(dtype)
    }
}

pub enum Integer {
    I8,
    I16,
    I32,
    I64,
    I128,
    ISIZE,
}

pub enum Unsigned {
    U8,
    U16,
    U32,
    U64,
    U128,
    USIZE,
}
