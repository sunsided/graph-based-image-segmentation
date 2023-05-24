use criterion::{criterion_group, criterion_main, Criterion};
use graph_based_image_segmentation::segmentation::{EuclideanRGB, NodeMergingThreshold, Segmentation};
use opencv::{
    core::{Size, BORDER_DEFAULT},
    imgcodecs::{imdecode, IMREAD_COLOR},
    imgproc::gaussian_blur,
    prelude::*,
};
use std::time::Duration;

fn criterion_benchmark(c: &mut Criterion) {
    let sigma = 0.8f64;
    let kernel_size = 5;
    let threshold = 10f32;
    let segment_size = 10;

    let tree = Mat::from_slice(include_bytes!("../../data/tree.jpg")).unwrap();
    let mut image = imdecode(&tree, IMREAD_COLOR).unwrap();
    image = blur_image(&mut image, sigma, kernel_size).unwrap();

    let mut group = c.benchmark_group("segmentation");
    group.measurement_time(Duration::from_secs(15));

    group.bench_function("segment_image 0.8 10", |b| {
        b.iter(|| {
            let mut segmenter = Segmentation::new(
                EuclideanRGB::default(),
                NodeMergingThreshold::new(threshold),
                segment_size,
            );
            segmenter.segment_image(&image);
        })
    });

    group.finish();
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

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
