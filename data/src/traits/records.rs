/*
    Appellation: records <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/

pub trait Record {}

pub trait Records {
    type Item: Record;
}
