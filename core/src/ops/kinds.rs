/*
    Appellation: kinds <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/

pub enum Op {}

pub enum CompareOp {
    Eq,
    Ge,
    Gt,
    Le,
    Lt,
    Ne,
}

pub enum BinaryOp {
    Add,
    Div,
    Maximum,
    Minimum,
    Mul,
    Sub,
}

pub trait BinaryOperation<T> {
    type Output;

    fn eval(&self, lhs: T, rhs: T) -> Self::Output;
}

pub trait UnaryOperation<T> {
    type Output;

    fn eval(&self, arg: T) -> Self::Output;
}