use crate::graph::{ImageEdge, ImageGraph};
use crate::segmentation::{Distance, Magic};
use opencv::core::Vec3b;
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
        self.graph = ImageGraph::new_with_nodes(node_count);

        for i in 0..height {
            for j in 0..width {
                let node_index = width * i + j;
                let mut node = self.graph.nodes.get_node_at_mut(node_index);

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

                    let weight = self.distance.distance(&node, &other);
                    let edge = ImageEdge::new(node_index, other_index, weight);

                    edges.push(edge);
                }

                // Test bottom neighbor.
                if j < width - 1 {
                    let other_index = width * i + (j + 1);
                    let other = self.graph.nodes.get_node_at(other_index);

                    let weight = self.distance.distance(&node, &other);
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

            let mut s_n = graph.nodes.find_node_component_at(edge.n);
            let mut s_m = graph.nodes.find_node_component_at(edge.m);

            // Are the nodes in different components?
            if s_m.id != s_n.id {
                let should_merge = self.magic.magic(&s_n, &s_m, &edge);
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
    pub fn enforce_minimum_segment_size(&self, m: usize) {
        unimplemented!()
    }

    /// Derive labels from the produced oversegmentation.
    ///
    /// # Returns
    ///
    /// Labels as an integer matrix.
    pub fn derive_labels(&self) -> Mat {
        unimplemented!()
    }
}
