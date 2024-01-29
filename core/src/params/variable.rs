/*
   Appellation: variable <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Variables
//!
//! ## Overview
//! Variables extend the functionality of the 'Parameter' by enabling mutability.
//!

pub struct Variable;

pub enum P {
    Param,
    Variable(Box<Self>),
}
