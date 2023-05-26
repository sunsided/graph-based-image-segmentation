use opencv::prelude::Mat;

/// A segmentation result.
pub struct SegmentationResult {
    /// The matrix of segmented pixels.
    pub segmentation: Mat,
    /// The number of connected components (segments).
    pub num_components: usize,
}
