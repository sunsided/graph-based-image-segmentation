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
}

impl<D, M> Segmentation<D, M>
where
    D: Distance,
    M: Magic,
{
    pub fn new(distance: D, magic: M) -> Self {
        Self {
            distance,
            magic,
            height: 0,
            width: 0,
            graph: ImageGraph::default(),
        }
    }

    /// Build the graph nased on the image, i.e. compute the weights
    /// between pixels using the underlying distance.
    ///
    /// # Arguments
    ///
    /// * `image` - The image to oversegment.
    pub fn build_graph(&mut self, image: &Mat) {
        let height = image.rows() as usize;
        let width = image.cols() as usize;
        let node_count = height * width;

        self.width = width;
        self.height = height;
        self.graph = ImageGraph::new_with_nodes(node_count);

        for i in 0..height {
            for j in 0..width {
                let node_index = width * i + j;
                let mut node = self.graph.nodes.get_node_at(node_index).borrow_mut();

                let bgr = image.at_2d::<Vec3b>(i as i32, j as i32).unwrap().0;
                node.b = bgr[0];
                node.g = bgr[1];
                node.r = bgr[2];

                // Initialize label
                node.l = node_index;
                node.id = node_index;
                node.n = 1;
            }
        }

        let mut edges = Vec::new();

        for i in 0..height {
            for j in 0..width {
                let node_index = width * i + j;
                let node = self.graph.nodes.get_node_at(node_index);

                // Test right neighbor.
                if i < height - 1 {
                    let other_index = width * (i + 1) + j;
                    let other = self.graph.nodes.get_node_at(other_index);

                    let weight = self.distance.distance(&node.borrow(), &other.borrow());
                    let edge = ImageEdge::new(node_index, other_index, weight);

                    edges.push(edge);
                }

                // Test bottom neighbor.
                if j < width - 1 {
                    let other_index = width * i + (j + 1);
                    let other = self.graph.nodes.get_node_at(other_index);

                    let weight = self.distance.distance(&node.borrow(), &other.borrow());
                    let edge = ImageEdge::new(node_index, other_index, weight);

                    edges.push(edge);
                }
            }
        }

        for edge in edges.into_iter() {
            self.graph.edges.add_edge(edge);
        }
    }

    /// Oversegment the given graph.
    pub fn oversegment_graph(&mut self) {
        let graph = &mut self.graph;
        assert_ne!(graph.num_edges(), 0);

        graph.edges.sort_edges();

        for e in 0..graph.num_edges() {
            let edge = graph.edges.get_edge_at(e % graph.num_edges());

            let s_n_idx = graph.nodes.find_node_component_at(edge.n);
            let s_m_idx = graph.nodes.find_node_component_at(edge.m);

            if s_n_idx == s_m_idx {
                continue;
            }

            let mut s_n = graph.nodes.get_node_at(s_n_idx).borrow_mut();
            let mut s_m = graph.nodes.get_node_at(s_m_idx).borrow_mut();

            // Are the nodes in different components?
            debug_assert_ne!(s_m.id, s_n.id);
            if s_m.id != s_n.id {
                let should_merge = self.magic.should_merge(&s_n, &s_m, &edge);
                if should_merge {
                    graph.merge(&mut s_n, &mut s_m, &edge);
                }
            }
        }
    }

    /// Enforces the given minimum segment size.
    ///
    /// # Arguments
    ///
    /// * `m` - Minimum segment size in pixels.
    pub fn enforce_minimum_segment_size(&mut self, m: usize) {
        let graph = &mut self.graph;
        assert_ne!(graph.num_nodes(), 0);

        for e in 0..graph.num_edges() {
            let edge = graph.edges.get_edge_at(e);

            let s_n_idx = graph.nodes.find_node_component_at(edge.n);
            let s_m_idx = graph.nodes.find_node_component_at(edge.m);

            if s_n_idx == s_m_idx {
                continue;
            }

            let mut s_n = graph.nodes.get_node_at(s_n_idx).borrow_mut();
            let mut s_m = graph.nodes.get_node_at(s_m_idx).borrow_mut();

            debug_assert_ne!(s_m.l, s_n.l);
            if s_m.l != s_n.l {
                let should_merge = s_n.n < m || s_m.n < m;
                if should_merge {
                    graph.merge(&mut s_n, &mut s_m, &edge);
                }
            }
        }
    }

    /// Derive labels from the produced oversegmentation.
    ///
    /// # Returns
    ///
    /// Labels as an integer matrix.
    pub fn derive_labels(&self) -> Mat {
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

                let s_node_idx = self.graph.nodes.find_node_component_at(n);
                let s_node = self.graph.nodes.get_node_at(s_node_idx).borrow();
                let label = s_node.id as i32;

                *(labels.at_2d_mut(i as i32, j as i32).unwrap()) = label;
            }
        }

        labels
    }
}
