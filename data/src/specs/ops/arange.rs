/*
   Appellation: arange <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use ndarray::prelude::{Array, Ix1, Ix2};
use num::traits::{Num, NumCast, ToPrimitive};
use num::traits::real::Real;
use std::ops;

/// [Arange] is a utilitarian trait that facilitates the creation of 
/// some structure using a range of equally spaced values.
pub trait Arange<T> {
    fn arange(args: impl Into<ArangeArgs<T>>) -> Self;
}

impl<T> Arange<T> for Vec<T>
where
    T: Copy + Num + NumCast,
{
    fn arange(args: impl Into<ArangeArgs<T>>) -> Self {
        let args = args.into();
        let n: usize = args
            .stop()
            .to_usize()
            .expect("Failed to convert 'stop' to a usize");
        (0..n)
            .map(|i| args.start() + args.step() * T::from(i).unwrap())
            .collect()
    }
}

impl<S, T> Arange<S> for Array<T, Ix1>
where
    S: Copy + Num + ToPrimitive,
    T: Copy + Num + NumCast,
{
    fn arange(args: impl Into<ArangeArgs<S>>) -> Self {
        let args = args.into();
        let n: usize = args
            .stop()
            .to_usize()
            .expect("Failed to convert 'stop' to a usize");
        let start = T::from(args.start()).unwrap();
        let step = T::from(args.step()).unwrap();

        Array::from_iter((0..n).map(|i| start + step * T::from(i).unwrap()))
    }
}

impl<S, T> Arange<S> for Array<T, Ix2>
where
    S: Copy + Num + ToPrimitive,
    T: Copy + Num + NumCast,
{
    fn arange(args: impl Into<ArangeArgs<S>>) -> Self {
        let args = args.into();
        let start = T::from(args.start()).unwrap();
        let step = T::from(args.step()).unwrap();
        let n: usize = args
            .stop()
            .to_usize()
            .expect("Failed to convert 'stop' to a usize");
        let f = |(i, _j)| start + step * T::from(i).unwrap();
        Array::from_shape_fn((n, 1), f)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ArangeArgs<T> {
    Arange { start: T, stop: T, step: T },
    Between { start: T, stop: T },
    Until { stop: T },
}

impl<T> ArangeArgs<T>
where
    T: Copy + Num,
{
    /// Returns the start value of the range.
    pub fn start(&self) -> T {
        match self {
            ArangeArgs::Arange { start, .. } => *start,
            ArangeArgs::Between { start, .. } => *start,
            ArangeArgs::Until { .. } => T::zero(),
        }
    }
    /// Returns the stop value of the range.
    pub fn stop(&self) -> T {
        match self {
            ArangeArgs::Arange { stop, .. } => *stop,
            ArangeArgs::Between { stop, .. } => *stop,
            ArangeArgs::Until { stop } => *stop,
        }
    }
    /// Returns the step value of the range.
    pub fn step(&self) -> T {
        match self {
            ArangeArgs::Arange { step, .. } => *step,
            ArangeArgs::Between { .. } => T::one(),
            ArangeArgs::Until { .. } => T::one(),
        }
    }
    /// Returns the number of steps between the given boundaries
    pub fn steps(&self) -> usize
    where
        T: Real,
    {
        match self {
            ArangeArgs::Arange { start, stop, step } => {
                let n = ((*stop - *start) / *step).ceil().to_usize().unwrap();
                n
            }
            ArangeArgs::Between { start, stop } => {
                let n = (*stop - *start).to_usize().unwrap();
                n
            }
            ArangeArgs::Until { stop } => {
                let n = stop.to_usize().unwrap();
                n
            }
        }
    }
}

impl<T> From<ops::Range<T>> for ArangeArgs<T> {
    fn from(args: ops::Range<T>) -> Self {
        ArangeArgs::Between {
            start: args.start,
            stop: args.end,
        }
    }
}

impl<T> From<ops::RangeFrom<T>> for ArangeArgs<T> {
    fn from(args: ops::RangeFrom<T>) -> Self {
        ArangeArgs::Until { stop: args.start }
    }
}

impl<T> From<(T, T, T)> for ArangeArgs<T> {
    fn from(args: (T, T, T)) -> Self {
        ArangeArgs::Arange {
            start: args.0,
            stop: args.1,
            step: args.2,
        }
    }
}

impl<T> From<[T; 3]> for ArangeArgs<T> where T: Copy {
    fn from(args: [T; 3]) -> Self {
        ArangeArgs::Arange {
            start: args[0],
            stop: args[1],
            step: args[2],
        }
    }
}

impl<T> From<(T, T)> for ArangeArgs<T> {
    fn from(args: (T, T)) -> Self {
        ArangeArgs::Between {
            start: args.0,
            stop: args.1,
        }
    }
}



impl<T> From<T> for ArangeArgs<T>
where
    T: Num,
{
    fn from(stop: T) -> Self {
        ArangeArgs::Until { stop }
    }
}
