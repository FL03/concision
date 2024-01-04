/*
   Appellation: elements <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use num::complex::Complex;
use num::traits::NumOps;

pub trait Element: NumOps + NumOps<Complex<Self>, Complex<Self>> + Sized {}
