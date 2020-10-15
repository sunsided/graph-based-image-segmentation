use crate::graph::{ImageEdge, ImageNode};

/// Represents an image graph, consisting of one node per pixel which are 4-connected.
#[derive(Debug, Clone, Default)]
pub struct ImageGraph {
    /// Number of components.
    k: usize,
    /// All nodes in this graph.
    pub nodes: Nodes,
    /// All edges in this graph.
    pub edges: Edges,
}

#[derive(Debug, Clone, Default)]
pub struct Nodes {
    nodes: Vec<ImageNode>,
}

#[derive(Debug, Clone, Default)]
pub struct Edges {
    edges: Vec<ImageEdge>,
}

impl ImageGraph {
    /// Constructs an image graph with the given exact number of nodes.
    ///
    /// # Arguments
    ///
    /// * `n` - The number of nodes to allocate.
    pub fn new_with_nodes(n: usize) -> Self {
        Self {
            k: n,
            nodes: Nodes::allocated(n),
            ..Self::default()
        }
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
    pub fn merge(&mut self, s_n: &mut ImageNode, s_m: &mut ImageNode, e: &ImageEdge) {
        s_m.l = s_n.id;

        // Update count.
        s_n.n += s_m.n;

        // Update maximum weight.
        s_n.max_w = s_n.max_w.max(s_m.max_w).max(e.w);

        // Update component count.
        self.k -= 1;
    }
}

impl Nodes {
    pub fn allocated(n: usize) -> Self {
        Self {
            nodes: vec![ImageNode::default(); n],
        }
    }

    /// Set the node of the given index.
    ///
    /// # Arguments
    ///
    /// * `n` - The index of the node.
    /// * `node` - The node to set.
    pub fn set_node(&mut self, n: usize, node: ImageNode) {
        assert!(n < self.nodes.len());
        self.nodes[n] = node;
    }

    /// Add a new node.
    ///
    /// # Arguments
    ///
    /// * `node` - The node to add.
    pub fn add_node(&mut self, node: ImageNode) {
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
    pub fn get_node_at(&self, n: usize) -> &ImageNode {
        assert!(n < self.nodes.len());
        &self.nodes[n]
    }

    /// Get a mutable reference to the n-th node.
    ///
    /// # Arguments
    ///
    /// * `n` - The index of the node.
    ///
    /// # Return
    ///
    /// The node at index `n`.
    #[inline(always)]
    pub fn get_node_at_mut(&mut self, n: usize) -> &mut ImageNode {
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
    /// * `n` - The node to find the component for.
    ///
    /// # Returns
    ///
    /// The node representing the found component.
    pub fn find_node_component(&self, n: &mut ImageNode) -> &ImageNode {
        // Get component of node n.
        let mut l = n.l;
        let mut id = n.id;

        while l != id {
            id = self.nodes[l].id;
            l = self.nodes[l].id;
        }

        let s = &self.nodes[l];
        assert_eq!(s.l, s.id);

        // Save latest component.
        n.l = s.id;
        s
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }
}

impl Edges {
    /// Add a new edge.
    ///
    /// # Arguments
    ///
    /// * `edge` - The edge to add.
    pub fn add_edge(&mut self, edge: ImageEdge) {
        self.edges.push(edge)
    }

    /// Add new edges.
    ///
    /// # Arguments
    ///
    /// * `edges` - The edges to add.
    pub fn add_edges<I>(&mut self, edges: I)
    where
        I: Iterator<Item = ImageEdge>,
    {
        for edge in edges.into_iter() {
            self.add_edge(edge);
        }
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
    pub fn get_edge_at(&self, n: usize) -> &ImageEdge {
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
    pub fn get_edge_mut(&mut self, n: usize) -> &mut ImageEdge {
        assert!(n < self.edges.len());
        &mut self.edges[n]
    }

    /// Sorts the edges by weight.
    pub fn sort_edges(&mut self) {
        self.edges.sort_by(|a, b| a.w.partial_cmp(&b.w).unwrap());
    }

    pub fn len(&self) -> usize {
        self.edges.len()
    }
}
