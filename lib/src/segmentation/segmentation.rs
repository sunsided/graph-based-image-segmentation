use crate::graph::{ImageEdge, ImageNode};
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
    /// The underlying distance to use.
    distance: D,
    /// The magic part of graph segmentation.
    magic: M,
    /// Number of components.
    k: usize,
    /// All nodes in this graph.
    nodes: Vec<ImageNode>,
    /// All edges in this graph.
    edges: Vec<ImageEdge>,
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
            k: 0,
            edges: Vec::new(),
            nodes: Vec::new(),
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

        self.k = node_count;
        self.nodes = vec![ImageNode::default(); node_count];

        for i in 0..height {
            for j in 0..width {
                let node_index = width * i + j;
                let mut node = self.get_node_at_mut(node_index);

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
                let node = self.get_node_at(node_index);

                // Test right neighbor.
                if i < height - 1 {
                    let other_index = width * (i + 1) + j;
                    let other = self.get_node_at(other_index);

                    let weight = self.distance.distance(&node, &other);
                    let edge = ImageEdge::new(node_index, other_index, weight);

                    edges.push(edge);
                }

                // Test bottom neighbor.
                if j < width - 1 {
                    let other_index = width * i + (j + 1);
                    let other = self.get_node_at(other_index);

                    let weight = self.distance.distance(&node, &other);
                    let edge = ImageEdge::new(node_index, other_index, weight);

                    edges.push(edge);
                }
            }
        }

        for edge in edges.into_iter() {
            self.add_edge(edge);
        }
    }

    /// Oversegment the given graph.
    pub fn oversegment_graph(&mut self) {
        let num_nodes = self.nodes.len();
        assert_ne!(num_nodes, 0);
        assert_ne!(self.edges.len(), 0);

        self.sort_edges();

        for e in 0..self.edges.len() {
            let edge_index = e % self.edges.len();
            let edge = self.edges.get(edge_index).unwrap();
            debug_assert!(edge.n < num_nodes);
            debug_assert!(edge.m < num_nodes);

            let mut s_n = self.find_node_component_at(edge.n);
            let mut s_m = self.find_node_component_at(edge.m);

            // Are the nodes in different components?
            if s_m.id != s_n.id {
                let should_merge = self.magic.magic(&s_n, &s_m, &edge);
                if should_merge {
                    self.merge(&mut s_n, &mut s_m, &edge);
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

    /// Get the number of nodes.
    ///
    /// # Return
    ///
    /// The number of nodes.
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Get the number of edges.
    ///
    /// # Return
    ///
    /// The number of edges.
    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }

    /// Get the number of connected components.
    ///
    /// # Return
    ///
    /// The number connected components.
    pub fn num_components(&self) -> usize {
        self.k
    }

    /// Merge two pixels (that is merge two nodes).
    ///
    /// # Arguments
    ///
    /// * `s_n` - The first node.
    /// * `s_m` - The second node.
    /// * `e` - The corresponding edge.
    ///
    /// # Remarks
    ///
    /// Depending on the used "Distance", some lines may be commented out
    /// to speed up the algorithm.
    fn merge(&mut self, s_n: &mut ImageNode, s_m: &mut ImageNode, e: &ImageEdge) {
        s_m.l = s_n.id;

        // Update count.
        s_n.n += s_m.n;

        // Update maximum weight.
        s_n.max_w = s_n.max_w.max(s_m.max_w).max(e.w);

        // Update component count.
        self.k -= 1;
    }

    /// Set the node of the given index.
    ///
    /// # Arguments
    ///
    /// * `n` - The index of the node.
    /// * `node` - The node to set.
    fn set_node(&mut self, n: usize, node: ImageNode) {
        assert!(n < self.nodes.len());
        self.nodes[n] = node;
    }

    /// Add a new node.
    ///
    /// # Arguments
    ///
    /// * `node` - The node to add.
    fn add_node(&mut self, node: ImageNode) {
        self.nodes.push(node)
    }

    /// Get a reference to the n-th node.
    ///
    /// # Arguments
    ///
    /// * `n` - The index of the node.
    ///
    /// # Return
    ///
    /// The node at index `n`.
    fn get_node_at(&self, n: usize) -> &ImageNode {
        assert!(n < self.nodes.len());
        &self.nodes[n]
    }

    /// Get a reference to the n-th node.
    ///
    /// # Arguments
    ///
    /// * `n` - The index of the node.
    ///
    /// # Return
    ///
    /// The node at index `n`.
    fn get_node_at_mut(&mut self, n: usize) -> &mut ImageNode {
        assert!(n < self.nodes.len());
        &mut self.nodes[n]
    }

    /// When two nodes get merged, the first node is assigned the id of the second
    /// node as label. By traversing this labeling, the current component of each
    /// node (that is, pixel) can easily be identified and the label can be updated
    /// for efficiency.
    ///
    /// # Arguments
    ///
    /// * `index` - The index of the node to find the component for.
    ///
    /// # Returns
    ///
    /// The node representing the found component.
    fn find_node_component_at(&mut self, index: usize) -> &mut ImageNode {
        let n = self.nodes.get_mut(index).unwrap();

        // Get component of node n.
        let mut l = n.l;
        let mut id = n.id;

        while l != id {
            let node = self.nodes.get_mut(l).unwrap();
            id = node.id;
            l = node.id;
        }

        let s = self.nodes.get_mut(l).unwrap();
        assert_eq!(s.l, s.id);

        // Save latest component.
        n.l = s.id;
        s
    }

    /// Add a new edge.
    ///
    /// # Arguments
    ///
    /// * `edge` - The edge to add.
    fn add_edge(&mut self, edge: ImageEdge) {
        self.edges.push(edge)
    }

    /// Gets a reference to the n-th edge.
    ///
    /// # Arguments
    ///
    /// * `n` - The index of the edge.
    ///
    /// # Return
    ///
    /// The edge at index `n`.
    fn get_edge_at(&self, n: usize) -> &ImageEdge {
        assert!(n < self.edges.len());
        &self.edges[n]
    }

    /// Gets a mutable reference to the n-th edge.
    ///
    /// # Arguments
    ///
    /// * `n` - The index of the edge.
    ///
    /// # Return
    ///
    /// The edge at index `n`.
    fn get_edge_mut(&mut self, n: usize) -> &mut ImageEdge {
        assert!(n < self.edges.len());
        &mut self.edges[n]
    }

    /// Sorts the edges by weight.
    fn sort_edges(&mut self) {
        self.edges.sort_by(|a, b| a.w.partial_cmp(&b.w).unwrap());
    }
}
