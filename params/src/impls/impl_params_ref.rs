/*
    Appellation: impl_params_ref <module>
    Created At: 2025.12.10:23:37:09
    Contrib: @FL03
*/
use crate::ParamsRef;
use ndarray::Dimension;

impl<A, D> ParamsRef<A, D> where D: Dimension {}
