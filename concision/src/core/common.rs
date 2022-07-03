/*
   Appellation: common
   Context:
   Creator: FL03 <jo3mccain@icloud.com>
   Description:
       ... Summary ...
*/

#[derive(Clone, Debug, Hash, PartialEq)]
pub enum Dates<Tz = chrono::Utc>
where
    Tz: chrono::TimeZone,
{
    Objective(bson::DateTime),
    Relative(chrono::DateTime<Tz>),
    Standard(i64),
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test() {
        let date: Dates = Dates::Standard(chrono::Utc::now().timestamp());
        println!("timestamp: {:#?}", &date);
        assert_eq!(date, date)
    }
}
