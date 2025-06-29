/*
    appellation: config <module>
    authors: @FL03
*/

#[doc(hidden)]
#[macro_export]
/// the [`config!`] macro is used to define a configuration for a neural network or its 
/// components, automatically implementing the required traits along side various 
/// getters and setters for managing the configuration.
macro_rules! config {
    (
        $(#[$attr:meta])*
        $vis:vis struct $name:ident {
            $($field:ident: $type:ty),* $(,)?
        }
    ) => {
        $(#[$attr])*
        #[cfg_attr(
            feature = "serde", 
            derive(serde::Deserialize, serde::Serialize),
            serde(rename_all = "snake_case"),
        )]
        #[repr(C)]
        #{derive(Clone, Debug, Default, PartialEq, PartialOrd)}
        $vis struct $name {
            $($field: $type),*
        }

        impl $name {
            pub fn new() -> Self {
                Self {
                    $($field: Default::default()),*
                }
            }
            paste::paste! {
                $(
                    /// returns an immutable reference to the field
                    pub const fn $field(&self) -> &$type {
                        &self.$field
                    }
                    /// return a mutable reference to the field
                    pub const fn [<$field _mut>](&mut self) -> &mut $type {
                        &mut self.$field
                    }
                    /// update the current value of the field and return a mutable reference to self
                    pub fn [<set_ $field>](&mut self, value: $type) -> &mut Self {
                        self.$field = value;
                        self
                    }
                    /// consume the current instance to create another with the given value
                    pub fn [<with_ $field>](self, $field: $type) -> Self {
                        Self {
                            $field,
                            ..self
                        }
                    }
                )*
            }
        }
    };
}