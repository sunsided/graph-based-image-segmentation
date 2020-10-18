use crate::graph::ImageNode;
use crate::segmentation::Distance;

/// Manhattan (i.e. L1) distance.
pub struct ManhattanRGB {}

unsafe impl Sync for ManhattanRGB {}
unsafe impl Send for ManhattanRGB {}

const NORMALIZATION_TERM: f32 = 255f32 + 255f32 + 255f32;

impl Default for ManhattanRGB {
    fn default() -> Self {
        Self {}
    }
}

impl Distance for ManhattanRGB {
    fn distance(&self, n: &ImageNode, m: &ImageNode) -> f32 {
        let dr = n.r as f32 - m.r as f32;
        let dg = n.g as f32 - m.g as f32;
        let db = n.b as f32 - m.b as f32;
        (dr.abs() + dg.abs() + db.abs()) / NORMALIZATION_TERM
    }
}
