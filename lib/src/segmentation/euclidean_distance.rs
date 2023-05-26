use crate::{Distance, ImageNodeColor};

/// Euclidean RGB distance.
///
/// ## Example
/// ```
/// use graph_based_image_segmentation::{Distance, EuclideanRGB, ImageNodeColor};
/// let distance = EuclideanRGB::default();
/// assert_eq!(distance.distance(&(0, 0, 0).into(), &(0, 0, 0).into()), 0.0);
/// assert_eq!(distance.distance(&(0, 0, 0).into(), &(0, 255, 0).into()), (1_f32/3.).sqrt());
/// assert_eq!(distance.distance(&(0, 0, 0).into(), &(0, 255, 255).into()), (2_f32/3.).sqrt());
/// assert_eq!(distance.distance(&(0, 0, 0).into(), &(255, 255, 255).into()), 1.0);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct EuclideanRGB {}

unsafe impl Sync for EuclideanRGB {}
unsafe impl Send for EuclideanRGB {}

const NORMALIZATION_TERM: f32 = 1.0 / 441.6729559300637f32; // (255f32 * 255f32 * 3f32).sqrt();

impl EuclideanRGB {
    #[inline(always)]
    pub fn distance(&self, n: &ImageNodeColor, m: &ImageNodeColor) -> f32 {
        let dr = n.r as isize - m.r as isize;
        let dg = n.g as isize - m.g as isize;
        let db = n.b as isize - m.b as isize;
        ((dr * dr + dg * dg + db * db) as f32).sqrt() * NORMALIZATION_TERM
    }
}

impl Default for EuclideanRGB {
    fn default() -> Self {
        Self {}
    }
}

impl Distance for EuclideanRGB {
    #[inline(always)]
    fn distance(&self, n: &ImageNodeColor, m: &ImageNodeColor) -> f32 {
        self.distance(n, m)
    }
}
