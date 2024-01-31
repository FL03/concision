/*
   Appellation: matmul <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/


pub trait Matmul {
    fn matmul(self, rhs: Self) -> Self;
}