/*
    Appellation: identity <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/

pub struct Id {
    id: String,
    name: String,
    timestamp: i32,
}

impl Id {
    pub fn new(id: String, name: String, timestamp: i32) -> Self {
        Self {
            id,
            name,
            timestamp,
        }
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn timestamp(&self) -> i32 {
        self.timestamp
    }
}
