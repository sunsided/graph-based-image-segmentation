use crate::ImageNodeColor;

/// Trait to be implemented by a concrete distance. The distance defines
/// how the weights between nodes in the image graph are computed. See the paper
/// by Felzenswalb and Huttenlocher for details.
pub trait Distance {
    /// Compute the distance given two nodes.
    ///
    /// # Arguments
    ///
    /// * `n` - The first node.
    /// * `m` - The second node.
    ///
    /// # Returns
    ///
    /// The distance between the two nodes.
    fn distance(&self, n: &ImageNodeColor, m: &ImageNodeColor) -> f32;
}
