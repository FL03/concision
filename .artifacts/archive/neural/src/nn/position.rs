/*
    Appellation: position <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/

pub struct Position {
    index: usize,
}

impl Position {
    pub fn new(index: usize) -> Self {
        Self { index }
    }

    pub fn index(&self) -> usize {
        self.index
    }
}

pub struct Scope {}
