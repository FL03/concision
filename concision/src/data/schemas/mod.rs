/*
   Appellation: schemas
   Context:
   Creator: FL03 <jo3mccain@icloud.com>
   Description:
       ... Summary ...
*/

pub trait SchemaSpec<Context, Data> {
    fn constructor(&self, context: Context, data: Data) -> Self
    where
        Self: Sized;
}
