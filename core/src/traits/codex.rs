/// [Decode] defines a standard interface for decoding data.
pub trait Decode<Rhs> {
    type Output;

    fn decode(&self, values: Rhs) -> Self::Output;
}

/// [Encode] defines a standard interface for encoding data.
pub trait Encode<Rhs> {
    type Output;

    fn encode(&self, values: Rhs) -> Self::Output;
}

pub trait Codex<A, B> {
    type Encoder: Encode<A, Output = B>;
    type Decoder: Decode<B, Output = A>;

    fn encode(&self, values: A) -> B;

    fn decode(&self, values: B) -> A;
}
