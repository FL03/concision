/*
    Appellation: optimizer <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/

pub struct OptimizerConfig {
    pub learning_rate: f32,
    pub momentum: f32,
    pub weight_decay: f32,
}

pub struct OptimizerBase<C> {
    pub config: C,
}
