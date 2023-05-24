use crate::graph::ImageNode;
use crate::segmentation::Distance;

/// Manhattan (i.e. L1) distance.
///
/// ## Example
/// ```
/// use graph_based_image_segmentation::graph::ImageNode;
/// use graph_based_image_segmentation::segmentation::{Distance, ManhattanRGB};
/// let a = ImageNode { b: 0, g: 0, r: 0, ..Default::default() };
/// let b = ImageNode { b: 0, g: 255, r: 255, ..Default::default() };
/// let distance = ManhattanRGB::default();
/// assert_eq!(distance.distance(&a, &b), 0.6666667);
/// ```
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
