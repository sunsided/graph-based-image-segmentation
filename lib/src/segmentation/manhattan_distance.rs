use crate::{Distance, ImageNodeColor};

/// Manhattan (i.e. L1) distance.
///
/// ## Example
/// ```
/// use graph_based_image_segmentation::{Distance, ImageNodeColor, ManhattanRGB};
/// let distance = ManhattanRGB::default();
/// assert_eq!(distance.distance(&(0, 0, 0).into(), &(0, 0, 0).into()), 0.0);
/// assert_eq!(distance.distance(&(0, 0, 0).into(), &(0, 255, 0).into()), 0.33333334);
/// assert_eq!(distance.distance(&(0, 0, 0).into(), &(0, 255, 255).into()), 0.6666667);
/// assert_eq!(distance.distance(&(0, 0, 0).into(), &(255, 255, 255).into()), 1.0);
/// ```
pub struct ManhattanRGB {}

unsafe impl Sync for ManhattanRGB {}
unsafe impl Send for ManhattanRGB {}

const NORMALIZATION_TERM: f32 = 1.0 / (255f32 * 3f32);

impl ManhattanRGB {
    #[inline(always)]
    pub fn distance(&self, n: &ImageNodeColor, m: &ImageNodeColor) -> f32 {
        let dr = n.r as isize - m.r as isize;
        let dg = n.g as isize - m.g as isize;
        let db = n.b as isize - m.b as isize;
        ((dr.abs() + dg.abs() + db.abs()) as f32) * NORMALIZATION_TERM
    }
}

impl Default for ManhattanRGB {
    fn default() -> Self {
        Self {}
    }
}

impl Distance for ManhattanRGB {
    #[inline(always)]
    fn distance(&self, n: &ImageNodeColor, m: &ImageNodeColor) -> f32 {
        self.distance(n, m)
    }
}
