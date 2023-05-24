use crate::graph::{ImageNode, ImageNodeColor};
use crate::segmentation::Distance;

/// Euclidean RGB distance.
///
/// ## Example
/// ```
/// use graph_based_image_segmentation::{Distance, EuclideanRGB, ImageNodeColor};
/// let a = ImageNodeColor::new_bgr(0, 0, 0);
/// let b = ImageNodeColor::new_bgr(255, 255, 255);
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
    #[inline(always)]
    fn distance(&self, n: &ImageNodeColor, m: &ImageNodeColor) -> f32 {
        let dr = n.r as f32 - m.r as f32;
        let dg = n.g as f32 - m.g as f32;
        let db = n.b as f32 - m.b as f32;
        (dr * dr + dg * dg + db * db).sqrt() / NORMALIZATION_TERM
    }
}
