/*
   Appellation: converters <module>
   Creator: FL03 <jo3mccain@icloud.com>
   Description:
       ... Summary ...
*/

pub trait Convertible<Actor, Conf, Data, Cont> {
    fn convert(&self, actor: Actor, config: Conf, data: Data, context: Cont) -> Self
    where
        Self: Sized;
}
