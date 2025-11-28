/*
    Appellation: similarity <module>
    Contrib: @FL03
*/
#[cfg(feature = "alloc")]
use alloc::vec::Vec;
use num_traits::{Float, FromPrimitive};

/// Calculate similarity between two patterns
pub fn calculate_pattern_similarity<T>(pattern1: &[[T; 3]], pattern2: &[[T; 3]]) -> T
where
    T: core::iter::Sum + Float + FromPrimitive,
{
    if pattern1.len() != pattern2.len() {
        return T::zero();
    }

    let mut total_diff = T::zero();
    for i in 0..pattern1.len() {
        for j in 0..3 {
            total_diff = total_diff + (pattern1[i][j] - pattern2[i][j]).abs();
        }
    }

    // Convert difference to similarity (1.0 = identical, 0.0 = completely different)
    let max_diff = T::from_usize(pattern1.len() * 3).unwrap();
    T::one() - (total_diff / max_diff)
}
/// Check if two patterns are similar enough to be considered duplicates
pub fn is_similar_pattern<T>(pattern1: &[[T; 3]], pattern2: &[[T; 3]]) -> bool
where
    T: core::iter::Sum + Float + FromPrimitive,
{
    if pattern1.len() != pattern2.len() {
        return false;
    }

    calculate_pattern_similarity(pattern1, pattern2) > T::from_f32(0.9).unwrap()
}

#[cfg(feature = "alloc")]
/// Extract common patterns from historical sequences
pub fn extract_patterns<T>(history: &[Vec<[T; 3]>], min_length: usize) -> Vec<Vec<[T; 3]>>
where
    T: core::iter::Sum + Float + FromPrimitive,
{
    let mut common_patterns = Vec::new();
    let mut pattern_scores = Vec::new();

    // Skip if not enough history
    if history.len() < 2 {
        return common_patterns;
    }

    // For each sequence in history
    for i in 0..history.len() {
        let seq = &history[i];

        // Skip sequences that are too short
        if seq.len() < min_length {
            continue;
        }

        // For each possible pattern start position
        for start in 0..=(seq.len() - min_length) {
            // For each possible pattern length
            for len in min_length..=(seq.len() - start) {
                let pattern = &seq[start..start + len];

                // Calculate how many times this pattern appears in other sequences
                let mut occurrence_count = 0;
                let mut similarity_score = T::zero();

                // Check other sequences
                for (j, other_seq) in history.iter().enumerate() {
                    if i == j {
                        continue; // Skip self
                    }

                    if other_seq.len() < pattern.len() {
                        continue;
                    }

                    // Sliding window comparison
                    for k in 0..=other_seq.len() - pattern.len() {
                        let window = &other_seq[k..k + pattern.len()];

                        // Calculate similarity
                        let similarity = calculate_pattern_similarity(pattern, window);
                        if similarity > T::from_f32(0.8).unwrap() {
                            occurrence_count += 1;
                            similarity_score = similarity_score + similarity;
                            break; // Count only once per sequence
                        }
                    }
                }

                // If pattern occurs in multiple sequences, consider it significant
                if occurrence_count >= history.len() / 3 {
                    // Clone pattern to owned data
                    let owned_pattern = pattern.to_vec();
                    common_patterns.push(owned_pattern);
                    pattern_scores.push(similarity_score);
                }
            }
        }
    }

    // Sort patterns by score and remove duplicates
    // (taking only the top N distinct patterns)
    let mut unique_patterns = Vec::new();
    let mut indices: Vec<usize> = (0..common_patterns.len()).collect();
    indices.sort_by(|&i, &j| {
        pattern_scores[j]
            .partial_cmp(&pattern_scores[i])
            .unwrap_or(core::cmp::Ordering::Equal)
    });

    for &idx in indices.iter().take(5) {
        // Take top 5 patterns
        let pattern = &common_patterns[idx];
        if !unique_patterns
            .iter()
            .any(|p: &Vec<[T; 3]>| is_similar_pattern(p, pattern))
        {
            unique_patterns.push(pattern.clone());
        }
    }

    unique_patterns
}
