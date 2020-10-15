use crate::graph::ImageNode;
use crate::segmentation::Distance;

/// Euclidean RGB distance.
#[derive(Debug, Clone, Copy)]
pub struct EuclideanRGB {
    /// Normalization term.
    d: f32, // TODO: Unused
}

impl Default for EuclideanRGB {
    fn default() -> Self {
        Self {
            d: (255f32 * 255f32 * 3f32).sqrt(),
        }
    }
}

impl Distance for EuclideanRGB {
    fn distance(&self, n: &ImageNode, m: &ImageNode) -> f32 {
        let dr = n.r as f32 - m.r as f32;
        let dg = n.g as f32 - m.g as f32;
        let db = n.b as f32 - m.b as f32;
        (dr * dr + dg * dg + db * db).sqrt()
    }
}
