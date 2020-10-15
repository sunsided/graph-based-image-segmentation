use crate::graph::{ImageEdge, ImageNode};
use crate::segmentation::Magic;

/// The original criterion described in
///
/// > D. Stutz, A. Hermans, B. Leibe.
/// > Superpixels: An Evaluation of the State-of-the-Art.
/// > Computer Vision and Image Understanding, 2018.
#[derive(Debug, Clone, Copy)]
pub struct MagicThreshold {
    /// The threshold.
    c: f32,
}

impl MagicThreshold {
    /// # Arguments
    ///
    /// * `c` - The threshold.
    pub fn new(c: f32) -> Self {
        Self { c }
    }
}

impl Magic for MagicThreshold {
    fn magic(&self, s_n: &ImageNode, s_m: &ImageNode, e: &ImageEdge) -> bool {
        let threshold = (s_n.max_w + self.c / s_n.n as f32).min(s_m.max_w + self.c / s_m.n as f32);
        e.w < threshold
    }
}
