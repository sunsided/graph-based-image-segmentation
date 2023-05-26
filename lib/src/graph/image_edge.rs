use std::cmp::Ordering;

/// Represents an edge between two pixels in an image.
///  Each edge is characterized by a weight and the adjacent nodes.
#[derive(Debug, Copy, Clone, Default)]
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

impl PartialEq for ImageEdge {
    fn eq(&self, other: &Self) -> bool {
        self.w.eq(&other.w) && self.n.eq(&other.n) && self.m.eq(&other.m)
    }

    fn ne(&self, other: &Self) -> bool {
        self.w.ne(&other.w) || self.n.ne(&other.n) || self.m.ne(&other.m)
    }
}

impl Eq for ImageEdge {}

impl Ord for ImageEdge {
    fn cmp(&self, other: &Self) -> Ordering {
        // Main sorting is by edge weight ascending.
        // In order to improve cache coherency during processing, we then sort by index.
        let ord_w = self.w.partial_cmp(&other.w).unwrap_or(Ordering::Equal);
        let ord_n = self.n.cmp(&other.n);
        let ord_m = self.m.cmp(&other.m);
        ord_w.then(ord_n).then(ord_m)
    }
}

impl PartialOrd for ImageEdge {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }

    fn lt(&self, other: &Self) -> bool {
        self.w < other.w || self.n < other.n || self.m < other.m
    }

    fn le(&self, other: &Self) -> bool {
        self.w <= other.w && self.n <= other.n && self.m <= other.m
    }

    fn gt(&self, other: &Self) -> bool {
        self.w > other.w || self.n > other.n || self.m > other.m
    }

    fn ge(&self, other: &Self) -> bool {
        self.w >= other.w && self.n >= other.n && self.m >= other.m
    }
}
