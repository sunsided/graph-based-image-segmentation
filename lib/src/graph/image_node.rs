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
    pub l: usize,
    /// Size of node after merging with other nodes.
    pub n: usize,
    /// ID of the node.
    pub id: usize,
    /// Maximum weight.
    pub max_w: f32,
}