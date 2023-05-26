use graph_based_image_segmentation::{EuclideanRGB, NodeMergingThreshold, Segmentation};
use opencv::core::{
    min_max_loc, no_array, Point, Scalar, Size, Vec3b, Vector, BORDER_DEFAULT, CV_32SC1, CV_8UC1,
    CV_8UC3,
};
use opencv::imgcodecs::{imread, imwrite, IMREAD_COLOR};
use opencv::imgproc::gaussian_blur;
use opencv::prelude::*;
use std::time::Instant;

fn main() {
    let mut image = imread("data/tree.jpg", IMREAD_COLOR).unwrap();

    // Apply smoothing to suppress digitization artifacts.
    image = blur_image(&mut image, 0.8f64, 5).unwrap();

    let threshold = 10f32;
    let segment_size = 10;
    let mut segmenter = Segmentation::new(
        EuclideanRGB::default(),
        NodeMergingThreshold::new(threshold),
        segment_size,
    );

    let start = Instant::now();
    let result = segmenter.segment_image(&image);
    let done = Instant::now();

    let duration = done - start;

    println!(
        "Image size:         {} × {} = {} pixels",
        image.cols(),
        image.rows(),
        image.cols() * image.rows()
    );
    println!("Num. segments:      {}", result.num_components);
    println!("Duration:           {} ms", duration.as_millis());

    let mut min = 0f64;
    let mut max = 0f64;
    let mut min_loc = Point::default();
    let mut max_loc = Point::default();
    min_max_loc(
        &result.segmentation,
        Some(&mut min),
        Some(&mut max),
        Some(&mut min_loc),
        Some(&mut max_loc),
        &no_array(),
    )
    .unwrap();

    let mut labels_out = Mat::default();
    result
        .segmentation
        .convert_to(&mut labels_out, CV_8UC1, 255f64 / max, 0f64)
        .unwrap();

    let mut labels_colored = Mat::default();
    opencv::imgproc::apply_color_map(
        &labels_out,
        &mut labels_colored,
        opencv::imgproc::COLORMAP_TURBO,
    )
    .unwrap();

    imwrite("labels.jpg", &labels_colored, &Vector::default()).unwrap();

    let contours = draw_contours(&image, &result.segmentation).unwrap();
    imwrite("contours.jpg", &contours, &Vector::default()).unwrap();
}

fn blur_image(image: &Mat, sigma: f64, size: usize) -> opencv::Result<Mat> {
    let mut blurred = Mat::default();
    gaussian_blur(
        &image,
        &mut blurred,
        Size::new(size as i32, size as i32),
        sigma,
        sigma,
        BORDER_DEFAULT,
    )?;
    Ok(blurred)
}

fn draw_contours(image: &Mat, labels: &Mat) -> opencv::Result<Mat> {
    assert_eq!(image.empty(), false);
    assert_eq!(image.channels(), 3);
    assert_eq!(image.rows(), labels.rows());
    assert_eq!(image.cols(), labels.cols());
    assert_eq!(labels.typ(), CV_32SC1);

    let mut contours =
        Mat::new_rows_cols_with_default(image.rows(), image.cols(), CV_8UC3, Scalar::all(0f64))?;
    let color = Vec3b::all(0); // black contours

    for i in 0..contours.rows() {
        let mut c_row = contours.row(i)?;
        let i_row = image.row(i)?;
        for j in 0..contours.cols() {
            if is_4connected_boundary_pixel(&labels, i, j)? {
                *(c_row.at_mut::<Vec3b>(j)?) = color;
            } else {
                *(c_row.at_mut::<Vec3b>(j)?) = *i_row.at::<Vec3b>(j)?;
            }
        }
    }

    Ok(contours)
}

/// Check if the given pixel is a boundary pixel in the given segmentation.
fn is_4connected_boundary_pixel(labels: &Mat, row: i32, col: i32) -> opencv::Result<bool> {
    let pixel_row = labels.row(row)?;
    let pixel = pixel_row.at::<i32>(col)?;

    let not_left = row > 0;
    let not_right = row < labels.rows() - 1;
    let not_top = col > 0;
    let not_bottom = col < labels.cols() - 1;

    if not_left && pixel != labels.at_2d::<i32>(row - 1, col)? {
        return Ok(true);
    }

    if not_right && pixel != labels.at_2d::<i32>(row + 1, col)? {
        return Ok(true);
    }

    if not_top && pixel != pixel_row.at::<i32>(col - 1)? {
        return Ok(true);
    }

    if not_bottom && pixel != pixel_row.at::<i32>(col + 1)? {
        return Ok(true);
    }

    return Ok(false);
}
