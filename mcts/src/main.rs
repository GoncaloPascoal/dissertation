use std::{
    cell::{RefCell, RefMut},
    collections::{BTreeMap, HashSet},
    rc::{Rc, Weak},
};

use rand::{self, Rng};
use tch::{kind, IndexOp, Tensor};

#[derive(Clone, Debug)]
enum Action {
    Swap((usize, usize)),
    Cnot((usize, usize)),
}

#[derive(Clone, Debug)]
struct State {
    gates: Vec<(usize, usize)>,
    node_to_qubit: Vec<usize>,
    qubit_to_node: Vec<usize>,
    locked_nodes: Vec<u32>,
    actions: Vec<Action>,
}

impl State {
    fn new(gates: Vec<(usize, usize)>, initial_mapping: Vec<usize>) -> Self {
        let node_count = initial_mapping.len();
        let mut qubit_to_node = vec![0; node_count];

        for (node, qubit) in initial_mapping.iter().enumerate() {
            qubit_to_node[*qubit] = node;
        }

        State {
            gates,
            node_to_qubit: initial_mapping,
            qubit_to_node,
            locked_nodes: vec![0; node_count],
            actions: Vec::new(),
        }
    }

    fn is_terminal(&self) -> bool {
        self.gates.is_empty()
    }
}

#[derive(Debug)]
struct Node {
    state: State,
    children: Vec<Option<NodeRef>>,
    parent: Option<WeakNodeRef>,
    parent_action: Option<usize>,
    reward: f64,
    q_values: Tensor,
    visit_count: Tensor,
}

type NodeRef = Rc<RefCell<Node>>;
type WeakNodeRef = Weak<RefCell<Node>>;

impl Node {
    fn new(
        state: State,
        num_actions: usize,
        parent: Option<WeakNodeRef>,
        parent_action: Option<usize>,
        reward: f64,
    ) -> Self {
        let mut children = Vec::with_capacity(num_actions);
        children.resize_with(num_actions, Default::default);

        Node {
            state,
            children,
            parent,
            parent_action,
            reward,
            q_values: Tensor::zeros(num_actions as i64, kind::FLOAT_CPU),
            visit_count: Tensor::zeros(num_actions as i64, kind::INT64_CPU),
        }
    }

    fn new_root(initial_state: State, num_actions: usize) -> Self {
        Self::new(initial_state, num_actions, None, None, 0.0)
    }

    fn num_actions(&self) -> usize {
        self.children.len()
    }

    fn is_terminal(&self) -> bool {
        self.state.is_terminal()
    }

    fn select_uct(&self, action_mask: Vec<bool>) -> usize {
        let mut rng = rand::thread_rng();

        let total_visits = self.visit_count.sum(kind::Kind::Int);
        let action_mask = Tensor::from_slice(
            action_mask
                .iter()
                .map(|&valid| if valid { 0.0 } else { f32::NEG_INFINITY })
                .collect::<Vec<f32>>()
                .as_slice(),
        );

        let uct = &self.q_values
            + &action_mask * (total_visits + 1e-3).sqrt() / (&self.visit_count + 1e-3);
        let max_indices = Tensor::where_(&uct.eq_tensor(&uct.max())).pop().unwrap();

        let idx = [rng.gen_range(0..max_indices.numel()) as i64];
        max_indices.int64_value(&idx) as usize
    }

    fn select_q(&self, action_mask: Vec<bool>) -> usize {
        let mut rng = rand::thread_rng();

        let action_mask = Tensor::from_slice(
            action_mask
                .iter()
                .map(|&valid| if valid { 0.0 } else { f32::NEG_INFINITY })
                .collect::<Vec<f32>>()
                .as_slice(),
        );

        let masked_q_values = &self.q_values + action_mask;
        let max_indices = Tensor::where_(&masked_q_values.eq_tensor(&masked_q_values.max()))
            .pop()
            .unwrap();

        let idx = [rng.gen_range(0..max_indices.numel()) as i64];
        max_indices.int64_value(&idx) as usize
    }

    fn update_q_value(&mut self, action: usize, reward: f64) {
        let action = action as i64;

        self.q_values.i(action).copy_(
            &((self.q_values.i(action) * self.visit_count.i(action) + reward)
                / (self.visit_count.i(action) + 1)),
        );
        self.visit_count
            .i(action)
            .copy_(&(self.visit_count.i(action) + 1));
    }
}

struct Mcts {
    coupling_map: BTreeMap<(usize, usize), f64>,
    root: NodeRef,
    search_iters: u32,
    discount_factor: f64,
    node_count: usize,
    edge_count: usize,
    num_actions: usize,
}

impl Mcts {
    fn new(
        coupling_map: BTreeMap<(usize, usize), f64>,
        initial_state: State,
        search_iters: u32,
    ) -> Self {
        let node_count = coupling_map
            .keys()
            .map(|k| if k.0 > k.1 { k.0 } else { k.1 })
            .max()
            .unwrap();
        let edge_count = coupling_map.len();
        let num_actions = edge_count + 1;

        Mcts {
            coupling_map,
            root: Rc::new(RefCell::new(Node::new_root(initial_state, num_actions))),
            search_iters,
            discount_factor: 0.95,
            node_count,
            edge_count,
            num_actions,
        }
    }

    fn commit_action(&self) -> usize {
        self.num_actions - 1
    }

    fn expand_child(
        &self,
        node_rc: &NodeRef,
        mut node: RefMut<'_, Node>,
        action: usize,
    ) -> NodeRef {
        let (next_state, reward) = self.step(&node.state, action);

        let child_rc = Rc::new(RefCell::new(Node::new(
            next_state,
            node.num_actions(),
            Some(Rc::downgrade(&node_rc)),
            Some(action),
            reward,
        )));
        node.children[action] = Some(child_rc.clone());

        child_rc
    }

    fn action_mask(&self, state: &State) -> Vec<bool> {
        let mut mask = vec![true; self.num_actions];

        for (i, &edge) in self.coupling_map.keys().enumerate() {
            if state.locked_nodes[edge.0] != 0 || state.locked_nodes[edge.1] != 0 {
                mask[i] = false;
            }
        }

        if state.locked_nodes.iter().all(|&n| n == 0) {
            mask[self.commit_action()] = false;
        }

        mask
    }

    fn step(&self, state: &State, action: usize) -> (State, f64) {
        let mut next_state = state.clone();
        let reward;

        if action < self.edge_count {
            let (node_a, node_b) = *self.coupling_map.keys().nth(action).unwrap();
            let (qubit_a, qubit_b) = (
                next_state.node_to_qubit[node_a],
                next_state.node_to_qubit[node_b],
            );

            next_state.node_to_qubit[node_a] = qubit_b;
            next_state.node_to_qubit[node_b] = qubit_a;

            next_state.qubit_to_node[qubit_a] = node_b;
            next_state.qubit_to_node[qubit_b] = node_a;

            next_state.locked_nodes[node_a] = 1;
            next_state.locked_nodes[node_b] = 1;

            next_state.actions.push(Action::Swap((node_a, node_b)));

            reward = -3.0;
        } else {
            for value in next_state.locked_nodes.iter_mut() {
                *value = 0;
            }

            let indices = self.schedulable_gates(&next_state);
            let mut gates = Vec::new();

            for (i, &gate) in next_state.gates.iter().enumerate() {
                if !indices.contains(&i) {
                    gates.push(gate);
                } else {
                    next_state.locked_nodes[gate.0] = 1;
                    next_state.locked_nodes[gate.1] = 1;

                    let nodes = (
                        next_state.qubit_to_node[gate.0],
                        next_state.qubit_to_node[gate.1],
                    );
                    next_state.actions.push(Action::Cnot(nodes));
                }
            }

            next_state.gates = gates;
            reward = indices.len() as f64;
        }

        (next_state, reward)
    }

    fn schedulable_gates(&self, state: &State) -> HashSet<usize> {
        let mut indices = HashSet::new();
        let mut seen_qubits = HashSet::new();

        for (i, &gate) in state.gates.iter().enumerate() {
            if !seen_qubits.contains(&gate.0) && !seen_qubits.contains(&gate.1) {
                let nodes = (state.qubit_to_node[gate.0], state.qubit_to_node[gate.1]);

                if state.locked_nodes[nodes.0] == 0 && state.locked_nodes[nodes.1] == 0 {
                    let sorted = if nodes.0 < nodes.1 {
                        nodes
                    } else {
                        (nodes.1, nodes.0)
                    };

                    if self.coupling_map.contains_key(&sorted) {
                        indices.insert(i);
                    }
                }
            }

            seen_qubits.insert(gate.0);
            seen_qubits.insert(gate.1);
        }

        indices
    }

    fn search(&mut self) {
        for _ in 0..self.search_iters {
            let mut node_rc = self.root.clone();
            let mut expanded = false;

            while !expanded {
                if (*node_rc).borrow().is_terminal() {
                    break;
                }

                (node_rc, expanded) = {
                    let node = (*node_rc).borrow_mut();
                    let action = node.select_uct(self.action_mask(&node.state));

                    match node.children[action] {
                        Some(ref child_rc) => (child_rc.clone(), false),
                        None => {
                            let child_rc = self.expand_child(&node_rc, node, action);
                            (child_rc, true)
                        }
                    }
                };
            }

            let mut total_reward = 0.0;
            loop {
                node_rc = {
                    let node = (*node_rc).borrow_mut();

                    if let Some(ref parent_weak) = node.parent {
                        total_reward += node.reward + self.discount_factor * total_reward;

                        let parent_rc = parent_weak.upgrade().unwrap();
                        {
                            let mut parent = (*parent_rc).borrow_mut();
                            parent.update_q_value(node.parent_action.unwrap(), total_reward);
                        }
                        parent_rc
                    } else {
                        break;
                    }
                };
            }
        }
    }

    fn act(&mut self) {
        let mut commit_performed = false;
        let commit_action = self.commit_action();

        while !commit_performed {
            self.search();

            self.root = {
                let root = (*self.root).borrow_mut();
                let action = root.select_q(self.action_mask(&root.state));
                let child = &root.children[action];

                commit_performed = action == commit_action;

                match child {
                    Some(child) => child.clone(),
                    None => self.expand_child(&self.root, root, action),
                }
            };

            let mut root = (*self.root).borrow_mut();
            root.parent = None;
            root.parent_action = None;
        }
    }

    fn perform_episode(&mut self) {
        while !(*self.root).borrow_mut().is_terminal() {
            self.act();
        }
    }
}

fn main() {
    let coupling_map = BTreeMap::from([((0, 1), 0.0), ((1, 2), 0.0), ((1, 3), 0.0), ((3, 4), 0.0)]);
    let remaining_gates = vec![(0, 2), (2, 4), (1, 2), (1, 0), (1, 3), (4, 1)];
    let initial_mapping = vec![0, 1, 2, 3, 4];

    let initial_state = State::new(remaining_gates, initial_mapping);
    let mut mcts = Mcts::new(coupling_map, initial_state, 64);

    mcts.perform_episode();
    println!("{:?}", &(mcts.root).borrow_mut().state.actions);
}
