/*
    Appellation: context <module>
    Created At: 2025.12.14:06:17:36
    Contrib: @FL03
*/

pub trait RawContext {
    private! {}
}

impl RawContext for () {
    seal! {}
}
