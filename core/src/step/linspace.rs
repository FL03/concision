/*
    Appellation: linspace <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use std::ops::{self, Range};

fn calculate_step<T: ops::Div<Output = T> + ops::Sub<Output = T>>(
    bounds: Range<T>,
    capacity: T,
) -> T {
    (bounds.end - bounds.start) / capacity
}

fn stepsize<T: ops::Div<Output = T> + ops::Sub<Output = T>>(from: T, to: T, capacity: T) -> T {
    (to - from) / capacity
}

fn _linspace(a: f64, b: f64, capacity: usize) -> Vec<f64> {
    let stepsize = stepsize(a, b, capacity as f64);
    let mut data = Vec::with_capacity(capacity);
    let mut current = a;
    for _ in 0..capacity {
        data.push(current);
        current += stepsize;
    }
    data
}

pub struct Linspace {
    bounds: Range<f64>,
    capacity: usize,
    data: Vec<f64>,
    stepsize: f64,
}

impl Linspace {
    pub fn new(bounds: Range<f64>, capacity: usize) -> Self {
        let stepsize = calculate_step(bounds.clone(), capacity as f64);
        let mut data = Vec::with_capacity(capacity);
        let mut current = bounds.start;
        for _ in 0..capacity {
            data.push(current);
            current += stepsize;
        }
        Self {
            bounds,
            capacity,
            data,
            stepsize,
        }
    }

    pub fn bounds(&self) -> &Range<f64> {
        &self.bounds
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn data(&self) -> &Vec<f64> {
        &self.data
    }

    pub fn stepsize(&self) -> f64 {
        self.stepsize
    }

    pub fn compute(&mut self) {
        let mut current = self.bounds().start;
        for i in 0..self.capacity() {
            self.data[i] = current;
            current += self.stepsize;
        }
    }

    pub fn update(&mut self) {
        self.stepsize = stepsize(self.bounds().start, self.bounds().end, self.capacity as f64);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linspace_simple() {
        let bounds = 0.0..10.0;
        let capacity = 10;

        let res = Linspace::new(bounds.clone(), capacity);
        assert_eq!(res.bounds(), &bounds);
        assert_eq!(res.capacity(), capacity);
        assert_eq!(
            res.data(),
            &vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        );
    }

    #[test]
    fn test_linspace() {
        let bounds = 0.0..5.0;
        let capacity = 10;

        let res = Linspace::new(bounds.clone(), capacity);
        assert_eq!(res.bounds(), &bounds);
        assert_eq!(res.capacity(), capacity);
        assert_eq!(
            res.data(),
            &vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
        );
    }

    #[test]
    fn test_linspace_large() {
        let bounds = 0.0..10.0;
        let capacity = 100;

        let res = Linspace::new(bounds.clone(), capacity);
        assert_eq!(res.bounds(), &bounds);
        assert_eq!(res.capacity(), capacity);
        assert_eq!(res.data()[1], 0.1);

        let bounds = 1.2..10.0;
        let capacity = 100;

        let res = Linspace::new(bounds.clone(), capacity);
        assert_eq!(res.bounds(), &bounds);
        assert_eq!(res.capacity(), capacity);
        assert_eq!(res.data()[0], 1.2);
        assert_eq!(res.data()[1], 1.288);
        let last = {
            let tmp = format!("{:.3}", res.data().last().unwrap());
            tmp.parse::<f64>().unwrap()
        };
        assert_eq!(last, 10.0 - res.stepsize());
    }
}
