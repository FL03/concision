/*
   Appellation: arithmetic <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use std::ops::{Add, Div, Mul, Sub};


pub trait Arithmetic<S, T = S> where Self: Add<S, Output = T> + Div<S, Output = T> + Mul<S, Output = T> + Sub<S, Output = T>  {

}

impl<A, S, T> Arithmetic<S, T> for A where A: Add<S, Output = T> + Div<S, Output = T> + Mul<S, Output = T> + Sub<S, Output = T>  {

}

pub trait Trig {
    fn cos(self) -> Self;

    fn cosh(self) -> Self;

    fn sin(self) -> Self;

    fn sinh(self) -> Self;

    fn tan(self) -> Self;

    fn tanh(self) -> Self;

    fn to_degrees(self) -> Self;

    fn to_radians(self) -> Self;
}