use graph_based_image_segmentation::segmentation::{EuclideanRGB, MagicThreshold, Segmentation};
use opencv::core::{Point, Scalar, Vector, no_array, min_max_loc, scale_add, CV_64F, CV_8UC1, CV_32SC1, CV_8UC3, Vec3b};
use opencv::imgcodecs::{imread, imwrite, IMREAD_COLOR};
use opencv::imgproc::cvt_color;
use opencv::prelude::*;
use std::time::Instant;

fn main() {
    let mut image = imread("medium.jpg", IMREAD_COLOR).unwrap();

    let threshold = 2000f32; // TODO: Revisit after distance normalization TODOs are addressed
    let mut segmenter = Segmentation::new(EuclideanRGB::default(), MagicThreshold::new(threshold));

    let start_bg = Instant::now();
    segmenter.build_graph(&image);
    let start_og = Instant::now();
    segmenter.oversegment_graph();
    let start_emss = Instant::now();
    segmenter.enforce_minimum_segment_size(10);
    let start_dl = Instant::now();
    let labels = segmenter.derive_labels();
    let done = Instant::now();

    let duration_bg = start_og - start_bg;
    let duration_og = start_emss - start_og;
    let duration_emss = start_dl - start_emss;
    let duration_dl = done - start_dl;
    let duration = done - start_bg;

    println!("Building the graph: {} ms", duration_bg.as_millis());
    println!("Oversegmentation:   {} ms", duration_og.as_millis());
    println!("Segment size:       {} ms", duration_emss.as_millis());
    println!("Label extraction:   {} ms", duration_dl.as_millis());
    println!("Total:              {} ms", duration.as_millis());

    let mut min = 0f64;
    let mut max = 0f64;
    let mut min_loc = Point::default();
    let mut max_loc = Point::default();
    min_max_loc(&labels, &mut min, &mut max, &mut min_loc, &mut max_loc, &no_array().unwrap());

    let mut labels_out = Mat::default().unwrap();
    labels.convert_to(&mut labels_out, CV_8UC1, 255f64/max, 0f64);
    imwrite("labels.jpg", &labels_out, &Vector::default()).unwrap();

    let contours = draw_contours(&image, &labels).unwrap();
    imwrite("contours.jpg", &contours, &Vector::default()).unwrap();
}

fn draw_contours(image: &Mat, labels: &Mat) -> opencv::Result<Mat> {
    assert_eq!(image.empty()?, false);
    assert_eq!(image.channels()?, 3);
    assert_eq!(image.rows(), labels.rows());
    assert_eq!(image.cols(), labels.cols());
    assert_eq!(labels.typ()?, CV_32SC1);

    let mut contours = Mat::new_rows_cols_with_default(image.rows(), image.cols(), CV_8UC3, Scalar::all(0f64))?;
    let color = Vec3b::all(0); // black contours

    for i in 0..contours.rows() {
        for j in 0..contours.cols() {
            if is_4connected_boundary_pixel(&labels, i, j)? {
                *contours.at_2d_mut::<Vec3b>(i, j)? = color;
            }
            else {
                *contours.at_2d_mut::<Vec3b>(i, j)? = *image.at_2d::<Vec3b>(i, j)?;
            }
        }
    }

    Ok(contours)
}

/// Check if the given pixel is a boundary pixel in the given segmentation.
fn is_4connected_boundary_pixel(labels: &Mat, row: i32, col: i32) -> opencv::Result<bool> {
    if row > 0 {
        if labels.at_2d::<i32>(row, col)? != labels.at_2d::<i32>(row - 1, col)? {
            return Ok(true);
        }
    }

    if row < labels.rows() - 1 {
        if labels.at_2d::<i32>(row, col)? != labels.at_2d::<i32>(row + 1, col)? {
            return Ok(true);
        }
    }

    if col > 0 {
        if labels.at_2d::<i32>(row, col)? != labels.at_2d::<i32>(row, col - 1)? {
            return Ok(true);
        }
    }

    if col < labels.cols() - 1 {
        if labels.at_2d::<i32>(row, col)? != labels.at_2d::<i32>(row, col + 1)? {
            return Ok(true);
        }
    }

    return Ok(false);
}