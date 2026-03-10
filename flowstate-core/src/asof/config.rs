/// Join direction for as-of matching.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AsOfDirection {
    /// Find rightmost right[j] <= left[i] (default for market data — no look-ahead).
    Backward,
    /// Find leftmost right[j] >= left[i].
    Forward,
    /// Find closest right[j] in either direction.
    Nearest,
}

/// Configuration for an as-of join operation.
#[derive(Debug, Clone)]
pub struct RustAsOfConfig {
    /// Maximum allowed distance in nanoseconds between matched timestamps.
    pub tolerance_ns: Option<i64>,
    /// Prefix to apply to right-side column names in the output.
    pub right_prefix: String,
    /// Join direction.
    pub direction: AsOfDirection,
    /// Whether to allow exact timestamp matches.
    pub allow_exact_match: bool,
}

impl Default for RustAsOfConfig {
    fn default() -> Self {
        Self {
            tolerance_ns: None,
            right_prefix: String::new(),
            direction: AsOfDirection::Backward,
            allow_exact_match: true,
        }
    }
}
