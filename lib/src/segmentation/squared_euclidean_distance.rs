use crate::{Distance, ImageNodeColor};

/// Squared Euclidean RGB distance.
///
/// ## Example
/// ```
/// use graph_based_image_segmentation::{Distance, SquaredEuclideanRGB};
/// let distance = SquaredEuclideanRGB::default();
/// assert_eq!(distance.distance(&(0, 0, 0).into(), &(0, 0, 0).into()), 0.0);
/// assert_eq!(distance.distance(&(0, 0, 0).into(), &(0, 255, 0).into()), (1_f32/3.));
/// assert_eq!(distance.distance(&(0, 0, 0).into(), &(0, 255, 255).into()), (2_f32/3.));
/// assert_eq!(distance.distance(&(0, 0, 0).into(), &(255, 255, 255).into()), 1.0);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct SquaredEuclideanRGB {}

unsafe impl Sync for SquaredEuclideanRGB {}
unsafe impl Send for SquaredEuclideanRGB {}

const NORMALIZATION_TERM: f32 = 1.0 / 195075.0; // (255f32 * 255f32 * 3f32);

impl SquaredEuclideanRGB {
    #[inline(always)]
    pub fn distance(&self, n: &ImageNodeColor, m: &ImageNodeColor) -> f32 {
        let dr = n.r as isize - m.r as isize;
        let dg = n.g as isize - m.g as isize;
        let db = n.b as isize - m.b as isize;
        ((dr * dr + dg * dg + db * db) as f32) * NORMALIZATION_TERM
    }
}

impl Default for SquaredEuclideanRGB {
    fn default() -> Self {
        Self {}
    }
}

impl Distance for SquaredEuclideanRGB {
    #[inline(always)]
    fn distance(&self, n: &ImageNodeColor, m: &ImageNodeColor) -> f32 {
        self.distance(n, m)
    }
}
