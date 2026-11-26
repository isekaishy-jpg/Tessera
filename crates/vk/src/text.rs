//! Text in fixed-size C `char` arrays (`char layerName[256]`).

use core::ffi::c_char;

/// The text in a fixed-size C `char` array up to its first NUL (all of it when there is none);
/// invalid UTF-8 becomes U+FFFD.
pub fn fixed_str(chars: &[c_char]) -> String {
    // `to_ne_bytes` takes the byte whether `c_char` is signed or not on the target.
    let bytes: Vec<u8> = chars
        .iter()
        .take_while(|&&c| c != 0)
        .map(|&c| c.to_ne_bytes()[0])
        .collect();
    // Valid text (the usual case) is taken without a second copy.
    String::from_utf8(bytes).unwrap_or_else(|e| String::from_utf8_lossy(e.as_bytes()).into_owned())
}
