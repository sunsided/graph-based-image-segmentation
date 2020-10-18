/// Represents a pixel in a video. Each pixel is represented by its
/// color which is needed to compute the weights between pixels.
#[derive(Debug, Clone, Default)]
pub struct ImageNode {
    /// Blue channel.
    pub b: u8,
    /// Green channel.
    pub g: u8,
    /// Red channel.
    pub r: u8,
    /// The label of the pixel (i.e. the index of the node this node belongs to).
    pub label: usize,
    /// Size of node after merging with other nodes.
    pub n: usize,
    /// ID of the node.
    pub id: usize,
    /// Maximum weight, i.e. the maximum distance in feature space
    /// of any two connected pixels of this set (see [ImageEdge]).
    ///
    /// [ImageEdge]: struct.ImageEdge.html#structfield.w
    pub max_w: f32,
}
