use criterion::{criterion_group, criterion_main, Criterion};
use graph_based_image_segmentation::segmentation::{Segmentation, EuclideanRGB, MagicThreshold};
use opencv::{imgproc::gaussian_blur, core::{Size, BORDER_DEFAULT}, prelude::*, imgcodecs::{IMREAD_COLOR, imdecode}};
use std::time::Duration;

fn criterion_benchmark(c: &mut Criterion) {
    let sigma = 0.8f64;
    let kernel_size = 5;
    let threshold = 2000f32; // TODO: Revisit after distance normalization TODOs are addressed

    let tree = Mat::from_slice(include_bytes!("../../data/tree.jpg")).unwrap();
    let mut image = imdecode(&tree, IMREAD_COLOR).unwrap();
    image = blur_image(&mut image, sigma, kernel_size).unwrap();

    let mut group = c.benchmark_group("segmentation");
    group.measurement_time(Duration::from_secs(15));

    group.bench_function("build_graph 0.8 2000", |b| b.iter(||
        {
            let mut segmenter = Segmentation::new(EuclideanRGB::default(), MagicThreshold::new(threshold));
            segmenter.build_graph(&image);
        }));

    group.bench_function("build_graph_and_oversegment 0.8 2000", |b| b.iter(||
        {
            let mut segmenter = Segmentation::new(EuclideanRGB::default(), MagicThreshold::new(threshold));
            segmenter.build_graph(&image);
            segmenter.oversegment_graph();
        }));

    group.finish();
}

fn blur_image(image: &Mat, sigma: f64, size: usize) -> opencv::Result<Mat> {
    let mut blurred = Mat::default()?;
    gaussian_blur(&image, &mut blurred, Size::new(size as i32, size as i32), sigma, sigma, BORDER_DEFAULT)?;
    Ok(blurred)
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
