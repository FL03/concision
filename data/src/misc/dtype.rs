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

impl From<i8> for DType {
    fn from(_: i8) -> Self {
        DType::Integer(Integer::I8)
    }
}

impl From<i16> for DType {
    fn from(_: i16) -> Self {
        DType::Integer(Integer::I16)
    }
}

impl From<i32> for DType {
    fn from(_: i32) -> Self {
        DType::Integer(Integer::I32)
    }
}

impl From<i64> for DType {
    fn from(_: i64) -> Self {
        DType::Integer(Integer::I64)
    }
}

impl From<i128> for DType {
    fn from(_: i128) -> Self {
        DType::Integer(Integer::I128)
    }
}

impl From<isize> for DType {
    fn from(_: isize) -> Self {
        DType::Integer(Integer::ISIZE)
    }
}

impl From<u8> for DType {
    fn from(_: u8) -> Self {
        DType::Unsigned(Unsigned::U8)
    }
}

impl From<u16> for DType {
    fn from(_: u16) -> Self {
        DType::Unsigned(Unsigned::U16)
    }
}

impl From<u32> for DType {
    fn from(_: u32) -> Self {
        DType::Unsigned(Unsigned::U32)
    }
}

impl From<u64> for DType {
    fn from(_: u64) -> Self {
        DType::Unsigned(Unsigned::U64)
    }
}

impl From<u128> for DType {
    fn from(_: u128) -> Self {
        DType::Unsigned(Unsigned::U128)
    }
}

impl From<usize> for DType {
    fn from(_: usize) -> Self {
        DType::Unsigned(Unsigned::USIZE)
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

impl From<i8> for Integer {
    fn from(_: i8) -> Self {
        Integer::I8
    }
}

impl From<i16> for Integer {
    fn from(_: i16) -> Self {
        Integer::I16
    }
}

impl From<i32> for Integer {
    fn from(_: i32) -> Self {
        Integer::I32
    }
}

impl From<i64> for Integer {
    fn from(_: i64) -> Self {
        Integer::I64
    }
}

impl From<i128> for Integer {
    fn from(_: i128) -> Self {
        Integer::I128
    }
}

impl From<isize> for Integer {
    fn from(_: isize) -> Self {
        Integer::ISIZE
    }
}

impl From<Integer> for DType {
    fn from(dtype: Integer) -> Self {
        DType::Integer(dtype)
    }
}

pub enum Unsigned {
    U8,
    U16,
    U32,
    U64,
    U128,
    USIZE,
}

impl From<u8> for Unsigned {
    fn from(_: u8) -> Self {
        Unsigned::U8
    }
}

impl From<u16> for Unsigned {
    fn from(_: u16) -> Self {
        Unsigned::U16
    }
}

impl From<u32> for Unsigned {
    fn from(_: u32) -> Self {
        Unsigned::U32
    }
}

impl From<u64> for Unsigned {
    fn from(_: u64) -> Self {
        Unsigned::U64
    }
}

impl From<u128> for Unsigned {
    fn from(_: u128) -> Self {
        Unsigned::U128
    }
}

impl From<usize> for Unsigned {
    fn from(_: usize) -> Self {
        Unsigned::USIZE
    }
}
