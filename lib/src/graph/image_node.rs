/// Represents a pixel in a video. Each pixel is represented by its
/// color which is needed to compute the weights between pixels.
#[derive(Debug, Copy, Clone, Default)]
#[repr(align(32))]
pub struct ImageNode {
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

/// Represents a pixel in a video. Each pixel is represented by its
/// color which is needed to compute the weights between pixels.
#[derive(Debug, Copy, Clone, Default)]
#[repr(align(4))]
pub struct ImageNodeColor {
    /// Blue channel.
    pub b: u8,
    /// Green channel.
    pub g: u8,
    /// Red channel.
    pub r: u8,
}

impl ImageNodeColor {
    #[inline(always)]
    pub const fn new_rgb(r: u8, g: u8, b: u8) -> Self {
        Self { b, g, r }
    }

    #[inline(always)]
    pub const fn new_bgr(b: u8, g: u8, r: u8) -> Self {
        Self { b, g, r }
    }
}

impl From<(u8, u8, u8)> for ImageNodeColor {
    fn from(value: (u8, u8, u8)) -> Self {
        ImageNodeColor::new_rgb(value.0, value.1, value.2)
    }
}

impl From<(f32, f32, f32)> for ImageNodeColor {
    fn from(value: (f32, f32, f32)) -> Self {
        ImageNodeColor::new_rgb(
            (value.0 * 255.0).clamp(0.0, 255.0) as u8,
            (value.1 * 255.0).clamp(0.0, 255.0) as u8,
            (value.2 * 255.0).clamp(0.0, 255.0) as u8,
        )
    }
}
