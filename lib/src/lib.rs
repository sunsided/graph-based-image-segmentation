//! # Efficient Graph-Based Image Segmentation
//!
//! This repository contains a Rust implementation of the graph-based image segmentation algorithms
//! described in \c[1] (available [here](http://cs.brown.edu/~pff/segment/))
//! focussing on generating over-segmentations, also referred to as superpixels.
//!
//! | Contours                 | Labels                 |
//! |--------------------------|------------------------|
//! | ![](https://github.com/sunsided/graph-based-image-segmentation/raw/842c931f9e301913a9101827ca880d1f03f27572/images/contours.jpg) | ![](https://github.com/sunsided/graph-based-image-segmentation/raw/842c931f9e301913a9101827ca880d1f03f27572/images/labels.jpg) |
//!
//! Please note that this is a reference implementation and not particularly fast.
//!
//! ```plain
//! [1] P. F. Felzenswalb and D. P. Huttenlocher.
//!     Efficient Graph-Based Image Segmentation.
//!     International Journal of Computer Vision, volume 59, number 2, 2004.
//! ```
//!
//! The implementation is based on [this work](https://github.com/davidstutz/graph-based-image-segmentation) by David Stutz,
//! which in turn was used in \[2] for evaluation.
//!
//! ```plain
//! [2] D. Stutz, A. Hermans, B. Leibe.
//!     Superpixels: An Evaluation of the State-of-the-Art.
//!     Computer Vision and Image Understanding, 2018.
//! ```
//!
//! ## Example use
//!
//! ```no_run
//! use opencv::imgcodecs::{imread, IMREAD_COLOR};
//! use graph_based_image_segmentation::{Segmentation, EuclideanRGB, NodeMergingThreshold};
//!
//! fn main() {
//!     let mut image = imread("data/tree.jpg", IMREAD_COLOR).unwrap();
//!
//!     let threshold = 10f32;
//!     let segment_size = 10;
//!     let mut segmenter = Segmentation::new(
//!         EuclideanRGB::default(),
//!         NodeMergingThreshold::new(threshold),
//!         segment_size,
//!     );
//!
//!     // NOTE: The image should be blurred before use; this is left out here for brevity.
//!     let labels = segmenter.segment_image(&image);
//! }
//! ```
mod graph;
mod segmentation;

pub use graph::ImageNodeColor;

pub use segmentation::{
    Distance, EuclideanRGB, ManhattanRGB, NodeMerging, NodeMergingThreshold, Segmentation,
    SegmentationResult, SquaredEuclideanRGB,
};
