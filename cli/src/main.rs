use graph_based_image_segmentation::segmentation::{EuclideanRGB, MagicThreshold, Segmentation};
use opencv::core::{Point, Scalar, Vector, no_array, min_max_loc, scale_add, CV_64F, CV_8UC1};
use opencv::imgcodecs::{imread, imwrite, IMREAD_COLOR};
use opencv::imgproc::{draw_contours, LINE_4, cvt_color};
use opencv::prelude::*;

fn main() {
    let mut image = imread("medium.jpg", IMREAD_COLOR).unwrap();

    let threshold = 2000f32; // TODO: Revisit after distance normalization TODOs are addressed
    let mut segmenter = Segmentation::new(EuclideanRGB::default(), MagicThreshold::new(threshold));
    segmenter.build_graph(&image);
    segmenter.oversegment_graph();
    segmenter.enforce_minimum_segment_size(10);
    let labels = segmenter.derive_labels();

    let mut min = 0f64;
    let mut max = 0f64;
    let mut min_loc = Point::default();
    let mut max_loc = Point::default();
    min_max_loc(&labels, &mut min, &mut max, &mut min_loc, &mut max_loc, &no_array().unwrap());

    let mut labels_out = Mat::default().unwrap();
    labels.convert_to(&mut labels_out, CV_8UC1, 255f64/max, 0f64);
    imwrite("labels.jpg", &labels_out, &Vector::default()).unwrap();

    let hierarchy = no_array().unwrap();
    draw_contours(
        &mut image,
        &labels,
        -1,
        Scalar::from([0f64, 0f64, 1f64, 1f64]),
        1,
        LINE_4,
        &hierarchy,
        1,
        Point::default(),
    ).unwrap();
    imwrite("contours.jpg", &image, &Vector::default()).unwrap();
}
