/*
    Appellation: id <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use uuid::Uuid;

#[cfg(feature = "rand")]
pub(crate) fn uuid() -> Uuid {
    uuid::Uuid::new_v4()
}

#[cfg(not(feature = "rand"))]
pub(crate) fn uuid() -> Uuid {
    // let uuid = Uuid::new_v6(&Uuid::NAMESPACE_DNS, b"rust-lang.org");
    Uuid::nil()
}

#[cfg(not(feature = "rand"))]
pub fn v5(data: impl AsRef<[u8]>) -> Uuid {
    let namespace = Uuid::NAMESPACE_OID;
    Uuid::new_v5(&namespace, data.as_ref())
}

#[cfg(not(feature = "rand"))]
pub fn v8(name: &str) -> Uuid {
    let mut buf = [0u8; 16];
    buf.copy_from_slice(name.as_bytes());
    Uuid::new_v8(buf)
}
