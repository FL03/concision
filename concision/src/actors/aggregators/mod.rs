/*
   Appellation: aggregators
   Context:
   Creator: FL03 <jo3mccain@icloud.com>
   Description:
       ... Summary ...
*/

pub enum Aggregators<Ac, Cnf, Cnt, Dt> {
    Dynamic(Box<dyn AggregatorSpec<Ac, Cnf, Cnt, Dt>>),
}

pub trait AggregatorSpec<Actor, Conf, Cont, Data> {
    fn configure(&self, config: Conf, controller: Cont) -> Actor
    where
        Self: Sized;
    fn curate(&self, actor: Actor) -> Self
    where
        Self: Sized;
}
