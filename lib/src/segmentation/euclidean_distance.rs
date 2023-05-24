use crate::graph::ImageNode;
use crate::segmentation::Distance;

/// Euclidean RGB distance.
///
/// ## Example
/// ```
/// use graph_based_image_segmentation::graph::ImageNode;
/// use graph_based_image_segmentation::segmentation::{Distance, EuclideanRGB};
/// let a = ImageNode { b: 0, g: 0, r: 0, ..Default::default() };
/// let b = ImageNode { b: 255, g: 255, r: 255, ..Default::default() };
/// let distance = EuclideanRGB::default();
/// assert_eq!(distance.distance(&a, &b), 1.0);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct EuclideanRGB {}

unsafe impl Sync for EuclideanRGB {}
unsafe impl Send for EuclideanRGB {}

const NORMALIZATION_TERM: f32 = 441.6729559300637f32; // (255f32 * 255f32 * 3f32).sqrt();

impl Default for EuclideanRGB {
    fn default() -> Self {
        Self {}
    }
}

impl Distance for EuclideanRGB {
    fn distance(&self, n: &ImageNode, m: &ImageNode) -> f32 {
        let dr = n.r as f32 - m.r as f32;
        let dg = n.g as f32 - m.g as f32;
        let db = n.b as f32 - m.b as f32;
        (dr * dr + dg * dg + db * db).sqrt() / NORMALIZATION_TERM
    }
}
