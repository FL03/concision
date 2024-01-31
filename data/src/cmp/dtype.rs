/*
   Appellation: dtype <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use serde::{Deserialize, Serialize};
use smart_default::SmartDefault;
use strum::{Display, EnumCount, EnumIs, EnumIter, EnumString, VariantNames};

pub trait DataType {
    fn dtype(&self) -> DType;
}

impl<T> DataType for T
where
    T: Copy + Into<DType>,
{
    fn dtype(&self) -> DType {
        (*self).into()
    }
}

#[derive(
    Clone,
    Copy,
    Debug,
    Deserialize,
    Display,
    EnumCount,
    EnumIs,
    EnumIter,
    EnumString,
    Eq,
    Hash,
    Ord,
    PartialEq,
    PartialOrd,
    Serialize,
    SmartDefault,
    VariantNames,
)]
#[serde(rename_all = "lowercase")]
#[strum(serialize_all = "lowercase")]
pub enum DType {
    #[default]
    Float(FloatingPoint),
    Integer(Integer),
    Unsigned(Unsigned),
}

impl DType {
    pub fn detect<T>(var: T) -> Self
    where
        T: Copy + Into<DType>,
    {
        var.dtype()
    }
}

impl From<f32> for DType {
    fn from(_: f32) -> Self {
        DType::Float(FloatingPoint::F32)
    }
}

impl From<f64> for DType {
    fn from(_: f64) -> Self {
        DType::Float(FloatingPoint::F64)
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

#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    Deserialize,
    Display,
    EnumCount,
    EnumIs,
    EnumIter,
    EnumString,
    Eq,
    Hash,
    Ord,
    PartialEq,
    PartialOrd,
    Serialize,
    VariantNames,
)]
#[serde(rename_all = "lowercase")]
#[strum(serialize_all = "lowercase")]
pub enum FloatingPoint {
    F32,
    #[default]
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
        DType::Float(dtype)
    }
}

pub struct Int {
    size: IntSize,
}

#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    Deserialize,
    Display,
    EnumCount,
    EnumIs,
    EnumIter,
    EnumString,
    Eq,
    Hash,
    Ord,
    PartialEq,
    PartialOrd,
    Serialize,
    VariantNames,
)]
#[serde(rename_all = "lowercase")]
#[strum(serialize_all = "lowercase")]
pub enum IntSize {
    #[default]
    S8 = 8,
    S16 = 16,
    S32 = 32,
    S64 = 64,
    S128 = 128,
    SSize,
}

#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    Deserialize,
    Display,
    EnumCount,
    EnumIs,
    EnumIter,
    EnumString,
    Eq,
    Hash,
    Ord,
    PartialEq,
    PartialOrd,
    Serialize,
    VariantNames,
)]
#[serde(rename_all = "lowercase")]
#[strum(serialize_all = "lowercase")]
pub enum Integer {
    I8,
    I16,
    I32,
    I64,
    I128,
    #[default]
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

#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    Deserialize,
    Display,
    EnumCount,
    EnumIs,
    EnumIter,
    EnumString,
    Eq,
    Hash,
    Ord,
    PartialEq,
    PartialOrd,
    Serialize,
    VariantNames,
)]
#[serde(rename_all = "lowercase")]
#[strum(serialize_all = "lowercase")]
pub enum Unsigned {
    U8,
    U16,
    U32,
    U64,
    U128,
    #[default]
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
