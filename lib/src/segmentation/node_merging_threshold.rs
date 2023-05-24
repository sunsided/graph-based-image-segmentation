use crate::graph::{ImageEdge, ImageNode};
use crate::segmentation::NodeMerging;

/// The original criterion described in
///
/// > D. Stutz, A. Hermans, B. Leibe.
/// > Superpixels: An Evaluation of the State-of-the-Art.
/// > Computer Vision and Image Understanding, 2018.
#[derive(Debug, Clone, Copy)]
pub struct NodeMergingThreshold {
    /// The threshold.
    c: f32,
}

impl NodeMergingThreshold {
    /// # Arguments
    ///
    /// * `c` - The threshold.
    pub fn new(c: f32) -> Self {
        Self { c }
    }
}

impl NodeMerging for NodeMergingThreshold {
    fn should_merge(&self, s_n: &ImageNode, s_m: &ImageNode, e: &ImageEdge) -> bool {
        let threshold_n = s_n.max_w + self.c / s_n.n as f32;
        let threshold_m = s_m.max_w + self.c / s_m.n as f32;

        // Edge weight muss be smaller than both thresholds.
        let threshold = threshold_n.min(threshold_m);
        e.w < threshold
    }
}
