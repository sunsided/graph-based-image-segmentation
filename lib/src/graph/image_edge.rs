/// Represents an edge between two pixels in an image.
///  Each edge is characterized by a weight and the adjacent nodes.
#[derive(Debug, PartialOrd, PartialEq, Clone, Default)]
pub struct ImageEdge {
    /// Index of first node.
    pub n: usize,
    /// Index of second node.
    pub m: usize,
    /// Edge weight, i.e. the distance of two pixels in feature space.
    pub w: f32,
}

impl ImageEdge {
    pub fn new(n: usize, m: usize, w: f32) -> Self {
        Self { n, m, w }
    }
}
