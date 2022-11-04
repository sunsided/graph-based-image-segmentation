use crate::graph::{ImageEdge, ImageGraph};
use crate::segmentation::{Distance, Magic};
use opencv::core::{Scalar, Vec3b, CV_32SC1};
use opencv::prelude::*;

/// Implementation of graph based image segmentation as described in the
/// paper by Felzenswalb and Huttenlocher.
#[derive(Debug)]
pub struct Segmentation<D, M>
where
    D: Distance,
    M: Magic,
{
    /// Image height.
    height: usize,
    /// Image width.
    width: usize,
    /// The constructed and segmented image graph.
    graph: ImageGraph,
    /// The underlying distance to use.
    distance: D,
    /// The magic part of graph segmentation.
    magic: M,
    /// The minimum size of the segments, in pixels.
    #[allow(dead_code)]
    segment_size: usize,
}

impl<D, M> Segmentation<D, M>
where
    D: Distance,
    M: Magic,
{
    pub fn new(distance: D, magic: M, segment_size: usize) -> Self {
        Self {
            distance,
            magic,
            height: 0,
            width: 0,
            segment_size,
            graph: ImageGraph::default(),
        }
    }

    /// Build the graph based on the image, i.e. compute the weights
    /// between pixels using the underlying distance.
    ///
    /// # Arguments
    ///
    /// * `image` - The image to oversegment.
    ///
    /// # Returns
    ///
    /// A matrix in `CV_32SC1` format containing the labels for each pixel.
    pub fn segment_image(&mut self, image: &Mat) -> Mat {
        self.build_graph(&image);
        self.oversegment_graph();
        self.enforce_minimum_segment_size(10);
        self.derive_labels()
    }

    /// Build the graph based on the image, i.e. compute the weights
    /// between pixels using the underlying distance.
    ///
    /// # Arguments
    ///
    /// * `image` - The image to oversegment.
    fn build_graph(&mut self, image: &Mat) {
        assert_eq!(image.empty(), false);
        self.height = image.rows() as usize;
        self.width = image.cols() as usize;
        self.graph = self.init_graph_nodes(&image);
        self.init_graph_edges();
    }

    /// Initializes the graph nodes from the image.
    fn init_graph_nodes(&mut self, image: &Mat) -> ImageGraph {
        debug_assert_ne!(self.height, 0);
        debug_assert_ne!(self.width, 0);
        let width = self.width;
        let height = self.height;
        let node_count = height * width;
        let graph = ImageGraph::new_with_nodes(node_count);

        for i in 0..height {
            for j in 0..width {
                let node_index = width * i + j;
                let mut node = graph.node_at(node_index).borrow_mut();

                let bgr = image.at_2d::<Vec3b>(i as i32, j as i32).unwrap().0;
                node.b = bgr[0];
                node.g = bgr[1];
                node.r = bgr[2];

                // Initialize label
                node.label = node_index;
                node.id = node_index;
                node.n = 1;
            }
        }

        graph
    }

    /// Initializes the edges between the nodes in the prepared graph.
    fn init_graph_edges(&mut self) {
        debug_assert_ne!(self.height, 0);
        debug_assert_ne!(self.width, 0);
        let height = self.height;
        let width = self.width;
        let graph = &mut self.graph;
        let distance = &self.distance;

        let mut edges = Vec::new();

        for i in 0..height {
            for j in 0..width {
                let node_index = width * i + j;
                let node = graph.node_at(node_index);

                // Test right neighbor.
                if j < width - 1 {
                    let other_index = width * i + (j + 1);
                    let other = graph.node_at(other_index);

                    let weight = distance.distance(&node.borrow(), &other.borrow());
                    let edge = ImageEdge::new(node_index, other_index, weight);

                    edges.push(edge);
                }

                // Test bottom neighbor.
                if i < height - 1 {
                    let other_index = width * (i + 1) + j;
                    let other = graph.node_at(other_index);

                    let weight = distance.distance(&node.borrow(), &other.borrow());
                    let edge = ImageEdge::new(node_index, other_index, weight);

                    edges.push(edge);
                }
            }
        }

        graph.clear_edges();
        graph.add_edges(edges.into_iter());
    }

    /// Oversegment the given graph.
    fn oversegment_graph(&mut self) {
        let graph = &mut self.graph;
        assert_ne!(graph.num_edges(), 0);

        graph.sort_edges();

        for e in 0..graph.num_edges() {
            debug_assert_eq!(e % graph.num_edges(), e);
            let edge_cell = graph.edge_at(e);

            // SAFETY: The edge is only borrow immutably here, and none of the
            //         node lookup methods on the graph operate on edges.
            //         Since each edge is only processed once, we can safely borrow "unsafely".
            let edge = unsafe { edge_cell.try_borrow_unguarded().unwrap() };

            let s_n_idx = graph.find_node_component_at(edge.n);
            let s_m_idx = graph.find_node_component_at(edge.m);

            if s_n_idx == s_m_idx {
                continue;
            }

            let mut s_n = graph.node_at(s_n_idx).borrow_mut();
            let mut s_m = graph.node_at(s_m_idx).borrow_mut();

            // Are the nodes in different components?
            debug_assert_ne!(s_m.id, s_n.id);
            let should_merge = self.magic.should_merge(&s_n, &s_m, &edge);
            if should_merge {
                graph.merge(&mut s_n, &mut s_m, &edge);
            }
        }
    }

    /// Enforces the given minimum segment size.
    ///
    /// # Arguments
    ///
    /// * `segment_size` - Minimum segment size in pixels.
    fn enforce_minimum_segment_size(&mut self, segment_size: usize) {
        let graph = &mut self.graph;
        assert_ne!(graph.num_nodes(), 0);

        for e in 0..graph.num_edges() {
            let edge = graph.edge_at(e).borrow();

            let s_n_idx = graph.find_node_component_at(edge.n);
            let s_m_idx = graph.find_node_component_at(edge.m);

            if s_n_idx == s_m_idx {
                continue;
            }

            let mut s_n = graph.node_at(s_n_idx).borrow_mut();
            let mut s_m = graph.node_at(s_m_idx).borrow_mut();

            // Neighboring segments must have different labels.
            debug_assert_ne!(s_m.label, s_n.label);

            let segment_too_small = s_n.n < segment_size || s_m.n < segment_size;
            if segment_too_small {
                graph.merge(&mut s_n, &mut s_m, &edge);
            }
        }
    }

    /// Derive labels from the produced oversegmentation.
    ///
    /// # Returns
    ///
    /// Labels as an integer matrix.
    fn derive_labels(&self) -> Mat {
        let mut labels = Mat::new_rows_cols_with_default(
            self.height as i32,
            self.width as i32,
            CV_32SC1,
            Scalar::from(0f64),
        )
        .unwrap();

        for i in 0..self.height {
            for j in 0..self.width {
                let n = self.width * i + j;

                let index = self.graph.find_node_component_at(n);
                let id = self.graph.node_id_at(index) as i32;

                *(labels.at_2d_mut(i as i32, j as i32).unwrap()) = id;
            }
        }

        labels
    }
}
