/*
    Appellation: point <module>
    Creator: FL03 <jo3mccain@icloud.com>
    Description:
        ... Summary ...
*/
use rand::{distributions::Standard, prelude::Distribution, Rng};
use std::ops::{Mul, Sub};

#[derive(Clone, Copy, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
pub enum Points {
    Origin(Point),
}

impl Default for Points {
    fn default() -> Self {
        Self::Origin(Point::new(0f64, 0f64, 0f64))
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct Point<T = f64> {
    pub x: T,
    pub y: T,
    pub z: T,
}

impl<T> Point<T> {
    pub fn new(x: T, y: T, z: T) -> Self {
        Self { x, y, z }
    }
    pub fn random_points(n: usize) -> Vec<Self>
        where
            Standard: Distribution<T>,
    {
        let mut res = Vec::with_capacity(n);
        for _ in 0..n {
            res.push(Self::default())
        }
        res
    }
}

impl<T> Default for Point<T>
    where
        Standard: Distribution<T>,
{
    fn default() -> Self {
        let mut rng = rand::thread_rng();
        Self::new(rng.gen::<T>(), rng.gen::<T>(), rng.gen::<T>())
    }
}

impl<T: std::fmt::Display> std::fmt::Display for Point<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {}, {})", self.x, self.y, self.z)
    }
}

impl<T: Mul + Mul<Output=T>> Mul for Point<T> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Self::new(self.x * rhs.x, self.y * rhs.y, self.z * rhs.z)
    }
}

impl<T: Sub + Sub<Output=T>> Sub for Point<T> {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_default() {
        let point = Point::<f64>::default();
        assert_eq!(&point, &point)
    }

    #[test]
    fn test_point_mul() {
        let f = |p: Point| p.clone() * p.clone();
        let point = Point::<f64>::default();
        assert_eq!(f(point.clone()), f(point.clone()))
    }

    #[test]
    fn test_point_sub() {
        let f = |p: Point| p.clone() - p.clone();
        let point = Point::<f64>::default();
        assert_eq!(f(point.clone()), Point::new(0f64, 0f64, 0f64))
    }

    #[test]
    fn test_random_points() {
        let points = Point::<f64>::random_points(100);
        assert_eq!(&points, &points)
    }
}
