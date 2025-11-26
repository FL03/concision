/*
    appellation: codex <module>
    authors: @FL03
*/
/// [Decode] defines a standard interface for decoding data.
pub trait Decode<Rhs> {
    type Output;

    fn decode(values: Rhs) -> Self::Output;
}

/// [Encode] defines a standard interface for encoding data.
pub trait Encode<Rhs> {
    type Output;

    fn encode(&self, values: Rhs) -> Self::Output;
}

pub trait Codex<A, B> {
    type Encoder<U, V>: Encode<U, Output = V>;
    type Decoder<U, V>: Decode<U, Output = V>;

    fn encode(&self) -> Self::Encoder<A, B>;

    fn decode() -> Self::Decoder<B, A>;
}
