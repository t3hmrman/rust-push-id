//! Firebase-style PushId generation
//!
//! This crate generates PushIds as specified/discussed in the [Firebase blog post](https://www.firebase.com/blog/2015-02-11-firebase-unique-identifiers.html)- introduced in 2015.
//!
//! Other implementations that served as inspiration:
//! - https://github.com/jengjeng/firebase-pushid-convert-timestamp
//! - https://github.com/alexdrone/PushID
//! - https://github.com/Darkwolf/node-pushid
//! - https://github.com/zerklabs/pushid

extern crate rand;

use rand::Rng;
use std::collections::HashMap;
use std::convert::TryInto;
use std::sync::OnceLock;

use std::time::{SystemTime, UNIX_EPOCH};

const PUSH_CHARS: &str = "-0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz";

static CHAR_INDEX_LOOKUP: OnceLock<HashMap<char, usize>> = OnceLock::new();
fn get_char_idx_lookup() -> &'static HashMap<char, usize> {
    CHAR_INDEX_LOOKUP.get_or_init(|| {
        PUSH_CHARS
            .chars()
            .enumerate()
            .fold(HashMap::new(), |mut acc, (idx, c)| {
                acc.insert(c, idx);
                acc
            })
    })
}

pub trait PushIdGen {
    fn get_id(&mut self) -> String;
}

pub struct PushId {
    /// Timestamp of the last Push
    last_time: u64,

    /// Bytes of randomness used to prevent collisions with other characters
    previous_indices: [usize; 12],
}

impl PushId {
    pub fn new() -> Self {
        let random_indices = PushId::generate_random_indices();
        PushId {
            last_time: 0,
            previous_indices: random_indices,
        }
    }

    fn gen_random_indices(&self, is_duplicate_time: bool) -> [usize; 12] {
        if is_duplicate_time {
            // If the timestamp hasn't changed since last push, use the same random number, except incremented by 1.
            let mut indices_copy = self.previous_indices.clone();

            for x in (0..12).rev() {
                if indices_copy[x] == 63 {
                    indices_copy[x] = 0;
                } else {
                    indices_copy[x] = indices_copy[x] + 1;
                    break;
                }
            }
            indices_copy
        } else {
            PushId::generate_random_indices()
        }
    }

    fn generate_random_indices() -> [usize; 12] {
        let mut rng = rand::thread_rng();
        let mut random_indices = [0; 12];
        for i in 0..12 {
            let n = rng.gen::<f64>() * 64 as f64;
            random_indices[i] = n as usize;
        }
        random_indices
    }

    fn gen_time_based_prefix(now: u64, mut acc: [usize; 8], i: u8) -> [usize; 8] {
        let index = (now % 64) as usize;
        acc[i as usize] = index;

        match now / 64 {
            new_now if new_now > 0 => PushId::gen_time_based_prefix(new_now, acc, i - 1),
            _ => acc, // We've reached the end of "time". Return the indices
        }
    }

    fn indices_to_characters(indices: Vec<&usize>) -> String {
        indices.iter().fold(String::from(""), |acc, &&x| {
            acc + &PUSH_CHARS
                .chars()
                .nth(x)
                .expect("Index out of range")
                .to_string()
        })
    }

    /// Retrieve the number of milliseconds since
    fn get_now() -> u64 {
        let start = SystemTime::now();
        let since_the_epoch = start
            .duration_since(UNIX_EPOCH)
            .expect("Unexpected time seed, EPOCH is not in the past");
        since_the_epoch.as_secs() * 1000 + since_the_epoch.subsec_nanos() as u64 / 1_000_000
    }

    /// Get the milliseconds since UNIX epoch for the PushId
    pub fn last_time_millis(&self) -> u64 {
        self.last_time.into()
    }

    /// Create a PushID from a given string
    pub fn from_str(s: impl AsRef<str>) -> Result<Self, std::io::Error> {
        let s = s.as_ref();
        if s.len() < 20 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("PushID [{s}] has invalid length"),
            ));
        }

        // Split into timestamp and randomness
        let (timestamp_bytes, randomness_bytes) = s.split_at(8);

        // Generate static lookup of characters to their values
        let char_idx_lookup = get_char_idx_lookup();

        // Convert timestamp bytes into a u64, by using the indices
        let timestamp_bytes_vec: Vec<u8> = Vec::new();
        for (idx, c) in timestamp_bytes.chars().rev().enumerate() {
            let value = match char_idx_lookup.get(&c) {
                Some(v) => v * (64 ^ idx), // TODO: THIS IS WRONG?
                None => {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        format!("failed to convert: {e}"),
                    ))
                }
            };
            timestamp_bytes_vec.push(value);
        }
        let timestamp_bytes: [u8; 8] = timestamp_bytes_vec.try_into().map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("failed to convert vec to [u8;8]"),
            )
        })?;
        let last_time = u64::from_le_bytes(timestamp_bytes);

        // Convert randomness into previous_indices member
        let other_indices = Vec::new();
        for c in other_indices.iter() {
            let idx = match char_idx_lookup.get(&c) {
                Some(v) => v * 64, // TODO: THIS IS WRONG
                None => {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        format!("failed to convert: {e}"),
                    ))
                }
            };
            other_indices.push(idx as usize);
        }
        let previous_indices: [usize; 12] = other_indices.try_into().map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("failed to remaining randomness into 12 bytes of indices"),
            )
        })?;

        Ok(Self {
            last_time,
            previous_indices,
        })
    }
}

impl PushIdGen for PushId {
    fn get_id(&mut self) -> String {
        let now = PushId::get_now();
        let is_duplicate_time = now == self.last_time;
        let prefix = PushId::gen_time_based_prefix(now, [0; 8], 7);
        let suffix = PushId::gen_random_indices(self, is_duplicate_time);
        self.previous_indices = suffix;
        self.last_time = PushId::get_now();
        let all = prefix.iter().chain(suffix.iter()).collect::<Vec<&usize>>();
        PushId::indices_to_characters(all)
    }
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use crate::{PushId, PushIdGen};

    /// Ensure that timestamps work properly
    #[test]
    fn test_timestamp() {
        let mut pushid = PushId::new();
        let id = pushid.get_id();
        assert!(!id.is_empty(), "generated pushid");

        let now = SystemTime::now();
        let millis_since = now
            .duration_since(UNIX_EPOCH)
            .expect("invalid epoch")
            .as_millis();
        let millis_since_pushid = pushid.last_time_millis() as u128;
        assert!(
            millis_since - millis_since_pushid < 10,
            "retrieved pushid generation time was within 10ms from now()"
        );
    }
}
