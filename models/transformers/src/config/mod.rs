/*
    Appellation: config <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/


pub struct TransformerConfig {
    pub heads: usize,
}

pub struct Features {
    
    pub d_model: usize,

}

pub struct QkvShape {
    pub dq: usize,
    pub dk: usize,
    pub dv: usize,
}

impl QkvShape {
    pub fn new(dq: usize, dk: usize, dv: usize) -> Self {
        Self {
            dq,
            dk,
            dv,
        }
    }
    
    pub fn std(dk: usize) -> Self {
        let (dq, dv) = (dk, dk);

        Self::new(dq, dk, dv)
    }
}


pub struct EmbedConfig {

}

pub struct FFNConfig {

}