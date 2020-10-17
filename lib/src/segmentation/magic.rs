use crate::graph::{ImageEdge, ImageNode};

/// The magic part of the graph segmentation, i.e. s given two nodes decide
/// whether to add an edge between them (i.e. merge the corresponding segments).
/// See the paper by Felzenswalb and Huttenlocher for details.
// TODO: Rename the trait.
pub trait Magic {
    /// Decide whether to merge the two segments corresponding to the given nodes or not.
    ///
    /// # Arguments
    ///
    /// * `s_n` - Node representing the first segment.
    /// * `s_m` - Node representing the second segment.
    /// * `e` - The edge between the two segments.
    ///
    /// # Returns
    ///
    /// `true` if merge
    // TODO: Rename the method.
    // TODO: Update the documentation on the return value.
    fn should_merge(&self, s_n: &ImageNode, s_m: &ImageNode, e: &ImageEdge) -> bool;
}
