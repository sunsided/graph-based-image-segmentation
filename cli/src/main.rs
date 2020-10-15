use graph_based_image_segmentation::segmentation::{EuclideanRGB, MagicThreshold, Segmentation};
use opencv::core::{Point, Scalar, Vector};
use opencv::imgcodecs::{imread, imwrite, IMREAD_COLOR};
use opencv::imgproc::{draw_contours, LINE_4};
use opencv::prelude::*;

fn main() {
    let mut image = imread("medium.jpg", IMREAD_COLOR).unwrap();

    let mut segmenter = Segmentation::new(EuclideanRGB::default(), MagicThreshold::new(1f32));
    segmenter.build_graph(&image);
    segmenter.oversegment_graph();
    segmenter.enforce_minimum_segment_size(10);
    let labels = segmenter.derive_labels();

    let hierarchy = Mat::default().unwrap();
    draw_contours(
        &mut image,
        &labels,
        -1,
        Scalar::from(1f64),
        1,
        LINE_4,
        &hierarchy,
        1,
        Point::default(),
    );
    imwrite("contours.jpg", &image, &Vector::default());
}
