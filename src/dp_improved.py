from vis_nav_game import Player, Action, Phase
import pygame
import cv2
import numpy as np
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pickle
import heapq
from tqdm import tqdm
from natsort import natsorted
import networkx as nx
import matplotlib.pyplot as plt
import logging
import matplotlib.pyplot as plt
from collections import deque
from collections import OrderedDict
import torch.nn.functional as F
import time
device = torch.device("cuda")

# image preprocessing/ transforms
preprocess = transforms.Compose(
[
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet means
        std=[0.229, 0.224, 0.225],
    ),
]
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class Autonnomous_resnet_Navigator(Player):
    # Configuration
    def __init__(self, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.save_dir = "../vis_nav_player/data/images_subsample"
        if not os.path.exists(self.save_dir):
            raise FileNotFoundError(f"Directory {self.save_dir} does not exist")
        if os.path.exists("codebook.pkl"):
            with open("codebook.pkl", "rb") as f:
                self.codebook = pickle.load(f)
        self.DATA_ROOT = "../vis_nav_player/data/"
        # --- Graph / Dijkstra path planning state ---
        # --- Graph / Dijkstra path planning state ---
        self.graph = None           # networkx graph
        self.current_path = None    # current planned path (list of image ids)
        self.path_valid_for_goal = None  # goal id for which current_path was computed
        self.planned_path = None    # list of node ids (for visualization)

        # Nodes we want the global planner to avoid (dead-ends, traps, etc.)
        self.blocked_nodes = set()

        #QUERY_DIR = os.path.join(DATA_ROOT, "query")
        self.DB_DIR = os.path.join(self.DATA_ROOT, "images_subsample")
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone = None
        self.fpv_fmap = None
        self.resnet_descriptors = None
        self.feature_database = None   # NEW: ResNet feature matrix [N, 512] (L2-normalized)

        self.turn_action =  Action.RIGHT # default turn action
        self.exploration_images = []
        self.stuck_cooldown = 0          # steps until we allow another stuck trigger
        self.best_dist_to_goal = None    # best (smallest) distance so far
        self.no_progress_steps = 0       # steps since last improvement
        self.db_index= None
        self.fpv = None
        self.goal_id = None
        
        self.current_id = None
        self.graph = None          # networkx graph
        self.planned_path = None   # list of node id
        self.num_images = 0
        self.nav_start_time = None
        self.time_budget = 60.0
        self.lazy_indexing = True
        self.position_history = deque(maxlen=20)
        self.action_history = deque(maxlen=10)  # Increased from 5 to 10
        self.consecutive_forward = 0
        self.consecutive_turns = 0  # NEW: Track consecutive turns (jittering)
        self.recovery_queue = deque()
        self.in_recovery_mode = False  # NEW: Track if we're executing recovery
        self._state = None
        self.show_visualization = False
        self.visualization_window = None
        self.target_comparison_window = None
        
        # NEW: Enhanced stuck detection
        self.last_positions = deque(maxlen=8)  # Track recent positions
        self.oscillation_counter = 0  # Count back-and-forth movements
        self.wall_collision_count = 0  # Count times we hit walls

        # NEW: Visited cell tracking to avoid loops
        self.visited_cells = {}  # {image_id: visit_count}
        self.visit_penalty_threshold = 3  # Strongly avoid cells visited 3+ times
        self.recent_visit_window = 50  # Only track visits within last 50 steps

        # Keep everything up to layer4 (excluding avgpool + fc)
        # children() = [conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool, fc]
        
        self.backbone_spatial = nn.Sequential(*list(resnet.children())[:-2]).to(self.DEVICE)
        self.backbone_spatial.to(self.device).eval()
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        self.preprocess_no_resize = transforms.Compose([
            transforms.ToTensor(),   # -> [3,H,W] float32 in [0,1]
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        super(Autonnomous_resnet_Navigator, self).__init__()

        if os.path.exists("resnet_descriptors.npy"):
            self.sift_descriptors = np.load("resnet_descriptors.npy")
        if os.path.exists("codebook.pkl"):
            with open("codebook.pkl", "rb") as f:
                self.codebook = pickle.load(f)

        logging.info("initialized")
    def plan_full_path(self, start_id: int, goal_id: int):
        """
        Use Dijkstra to compute the full shortest path from start_id to goal_id.
        Stores the path in self.planned_path and returns it.
        """
        if self.graph is None:
            raise RuntimeError("Graph is not built. Call build_topology_graph() first.")

        if start_id not in self.graph or goal_id not in self.graph:
            raise ValueError(f"start_id {start_id} or goal_id {goal_id} not in graph.")

        # NetworkX Dijkstra shortest path by edge weight
        path = nx.dijkstra_path(self.graph, source=start_id, target=goal_id, weight="weight")
        self.planned_path = path

        logging.info(f"Planned path from {start_id} to {goal_id} with {len(path)} nodes.")
        return path
    def save_path_to_file(self, filepath: str = "planned_path.txt"):
        """
        Save the current planned path as a simple text file like:
        0 -> 1 -> 2 -> 5 -> 9 ...
        """
        if not self.planned_path:
            logging.warning("No planned path to save.")
            return

        with open(filepath, "w") as f:
            f.write(" -> ".join(str(n) for n in self.planned_path))

        logging.info(f"Saved planned path to {filepath}")

    def save_graph_with_path_image(
        self,
        filename: str = "nav_graph.png",
        start_id: int = None,
        goal_id: int = None
        ):
        """
        Render the graph to an image, highlighting the planned path.
        - Nodes: small dots
        - Edges: light lines
        - Path edges: thick, dark line
        - Start node: green
        - Goal node: red
        """
        if self.graph is None:
            raise RuntimeError("Graph is not built. Call build_topology_graph() first.")

        G = self.graph

        if self.planned_path is None or len(self.planned_path) < 2:
            logging.warning("No non-trivial planned path; drawing graph without highlight.")
            path_edges = []
        else:
            # Consecutive node pairs along path
            path_edges = list(zip(self.planned_path[:-1], self.planned_path[1:]))

        # If not explicitly passed, deduce from path
        if start_id is None and self.planned_path:
            start_id = self.planned_path[0]
        if goal_id is None and self.planned_path:
            goal_id = self.planned_path[-1]

        # Layout: automatic 2D positions for nodes
        pos = nx.spring_layout(G, seed=42)  # deterministic layout

        plt.figure(figsize=(8, 6))

        # Draw all nodes
        nx.draw_networkx_nodes(G, pos, node_size=20)

        # Draw all edges lightly
        nx.draw_networkx_edges(G, pos, alpha=0.3, width=1.0)

        # Highlight path edges
        if path_edges:
            nx.draw_networkx_edges(
                G, pos,
                edgelist=path_edges,
                width=3.0,   # thicker
                edge_color="black"
            )

        # Highlight start / goal nodes, if available
        if start_id is not None and start_id in G:
            nx.draw_networkx_nodes(G, pos, nodelist=[start_id], node_size=60, node_color="green")
        if goal_id is not None and goal_id in G:
            nx.draw_networkx_nodes(G, pos, nodelist=[goal_id], node_size=60, node_color="red")

        plt.axis("off")
        plt.tight_layout()
        plt.savefig(filename, dpi=200)
        plt.close()

        logging.info(f"Saved graph visualization with path to {filename}")

    def build_topology_graph(self):
        """
        Simple example: chain graph over all image indices.
        Replace/extend this if you already have a neighbor structure.
        """
        G = nx.Graph()
        N = self.num_images

        for i in range(N):
            G.add_node(i)

        # Sequential neighbor connections; weight can be 1.0 or something smarter
        for i in range(N - 1):
            G.add_edge(i, i + 1, weight=1.0)

        self.graph = G
        logging.info(f"Built graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")



    import networkx as nx

    def dijkstra_shortest_path(self, start_id: int, goal_id: int):
        """
        Run Dijkstra on self.graph (a networkx.Graph) from start_id to goal_id.

        Returns a list of node indices [start_id, ..., goal_id] or None if
        the goal is unreachable.
        """
        if self.graph is None:
            self.build_topology_graph()
        if self.graph is None:
            logging.warning("dijkstra_shortest_path: graph is None after build.")
            return None

        G = self.graph

        if not G.has_node(start_id) or not G.has_node(goal_id):
            logging.warning(
                f"Dijkstra called with invalid ids: start={start_id}, "
                f"goal={goal_id}, nodes={list(G.nodes)[:5]}..."
            )
            return None

        blocked = getattr(self, "blocked_nodes", set())

        # Standard Dijkstra with a heap, but skip blocked nodes
        dist = {n: float("inf") for n in G.nodes}
        prev = {n: None for n in G.nodes}
        visited = set()

        dist[start_id] = 0.0
        heap = [(0.0, start_id)]

        while heap:
            d, u = heapq.heappop(heap)
            if u in visited:
                continue

            if u in blocked and u not in (start_id, goal_id):
                # Treat blocked nodes as removed from the graph
                continue

            visited.add(u)
            if u == goal_id:
                break

            # G[u] is a dict: neighbor -> attr_dict
            for v, attr in G[u].items():
                if v in blocked and v not in (start_id, goal_id):
                    continue

                w = float(attr.get("weight", 1.0))
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(heap, (nd, v))

        if goal_id not in visited or dist[goal_id] == float("inf"):
            logging.info(
                f"Dijkstra could not reach goal {goal_id} from start {start_id}; "
                f"blocked_nodes={blocked}"
            )
            return None

        # Reconstruct path
        path = []
        u = goal_id
        while u is not None:
            path.append(u)
            u = prev[u]
        path.reverse()
        return path

    def get_next_target_id_from_dijkstra(self, current_id, goal_id):
        # If no path or we've deviated too far, recompute
        if self.current_path is None or goal_id != self.goal_id:
            self.current_path = self.dijkstra_shortest_path(current_id, goal_id)

        if self.current_path is None or len(self.current_path) < 2:
            # Fall back to simple local heuristic if graph fails
            return current_id

        # Find current_id in path and move one step forward along it
        if current_id in self.current_path:
            idx = self.current_path.index(current_id)
            if idx < len(self.current_path) - 1:
                return self.current_path[idx + 1]
            else:
                return current_id
        else:
            # We fell off the path – lightweight replan
            self.current_path = self.dijkstra_shortest_path(current_id, goal_id)
            if self.current_path is None or len(self.current_path) < 2:
                return current_id
            return self.current_path[1]

    def find_goal(self):
        targets = self.get_target_images()
        if targets is None or len(targets) == 0:
            logging.error("No target images available")
            return None

        goal_candidates = []
        matched_images = []
        for i, target in enumerate(targets):
            indices, distances = self.get_neighbor(target, k=1)
            goal_candidates.append((indices[0], distances[0]))
            matched_images.append(self.exploration_images[indices[0]])
            logging.info(f"Target view {i}: ID {indices[0]} (dist: {distances[0]:.4f})")

        goal_id = goal_candidates[0][0]
        logging.info(f"Primary goal: Image {goal_id}")
        if self.show_visualization:
            self.show_target_comparison(targets, matched_images, goal_candidates)

        return goal_id
    def show_target_comparison(self, targets, matched_images, goal_info):
        """Display target images vs matched database images"""
        view_names = ['Front', 'Right', 'Back', 'Left']

        rows = []
        for i in range(4):
            target_resized = cv2.resize(targets[i], (320, 240))
            matched_resized = cv2.resize(matched_images[i], (320, 240))

            target_labeled = target_resized.copy()
            matched_labeled = matched_resized.copy()

            cv2.putText(target_labeled, f"{view_names[i]} Target", (10, 30),
                        cv2.FONT_ITALIC, 0.7, (255, 255, 255), 2)
            cv2.putText(matched_labeled, f"Matched: ID {goal_info[i][0]}", (10, 30),
                        cv2.FONT_ITALIC, 0.7, (255, 255, 255), 2)
            cv2.putText(matched_labeled, f"Dist: {goal_info[i][1]:.4f}", (10, 60),
                        cv2.FONT_ITALIC, 0.6, (255, 255, 255), 2)
            row = np.hstack([target_labeled, matched_labeled])
            rows.append(row)

        comparison = np.vstack(rows)

        cv2.imshow('Target Comparison', comparison)
        cv2.waitKey(1)
    # NEW: Check if path ahead is clear
    def is_path_clear(self, img, min_clear_distance=35):  # Balanced (was 50, too sensitive)
        """
        Check if there's enough clear space ahead to move forward
        Returns: (is_clear, forward_distance)
        """
        H, W = img.shape[:2]
        # Focus on center column of bottom half
        roi = img[H//2:, W//3:2*W//3]  # center third, bottom half
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        edges = cv2.Canny(roi_gray, 50, 150)
        
        # Check center columns for obstacles
        center_cols = edges[:, edges.shape[1]//3:2*edges.shape[1]//3]
        
        # Search from bottom up
        forward_clear = 0
        for row_idx in range(center_cols.shape[0] - 1, -1, -1):
            if np.any(center_cols[row_idx, :] > 0):
                forward_clear = center_cols.shape[0] - row_idx
                break
        else:
            forward_clear = center_cols.shape[0]
        
        return forward_clear > min_clear_distance, forward_clear
    
    # IMPROVED: Better stuck detection (MUCH LESS SENSITIVE)
    def is_stuck(self, dist_to_goal):
        """Enhanced stuck detection with multiple criteria"""
        # Need MORE history before checking
        if len(self.position_history) < 15:  # INCREASED from 8
            return False

        recent = list(self.position_history)[-15:]  # Look at more history
        unique = len(set(recent))

        # Criterion 1: Position oscillation (bouncing between few positions)
        from collections import Counter
        counts = Counter(recent)
        most_common_id, most_common_count = counts.most_common(1)[0]
        freq = most_common_count / len(recent)
        position_stuck = (unique <= 2 and freq > 0.6)  # STRICTER: only 2 unique positions

        # Criterion 2: Action oscillation (turning back and forth) - REMOVED, too sensitive
        action_stuck = False

        # Criterion 3: No progress toward goal
        if self.best_dist_to_goal is None or dist_to_goal < self.best_dist_to_goal - 2:  # Allow more variance
            self.best_dist_to_goal = dist_to_goal
            self.no_progress_steps = 0
        else:
            self.no_progress_steps += 1

        progress_stuck = self.no_progress_steps > 30  # MUCH HIGHER threshold (was 12)

        # Return True ONLY if BOTH criteria are met (much less sensitive)
        return position_stuck and progress_stuck
    def build_db_index(self):
        """
        Build an index of (filename, descriptor) for the gallery DB.

        If self.feature_database is already built (typical when running
        navigation / pre_navigation), we simply reuse those descriptors,
        assuming DB_DIR and save_dir point to the same image set and
        ordering uses natsorted.

        If self.feature_database is None (e.g., using this class just
        as a retrieval tool from main() without pre_navigation), we
        compute descriptors from scratch with compute_descriptor.
        """
        db_index = []

        # IMPORTANT: use natsorted so ordering is consistent with pre_navigation
        files = natsorted(
            [f for f in os.listdir(self.DB_DIR) if f.lower().endswith(".jpg")]
        )

        # Case 1: we already have a feature database built in pre_navigation
        if self.feature_database is not None:
            # self.feature_database: [N, 512], L2-normalized
            # pre_navigation also used natsorted over the same directory:
            #   files = natsorted([...])
            # so index i in feature_database corresponds to files[i]
            num_feats = self.feature_database.shape[0]
            if num_feats != len(files):
                logging.warning(
                    f"feature_database size ({num_feats}) does not match DB files ({len(files)}). "
                    f"Falling back to recomputing descriptors for DB."
                )
            else:
                for i, fname in enumerate(files):
                    desc = self.feature_database[i]
                    db_index.append((fname, desc.astype(np.float32)))
                return db_index  # done

        # Case 2: no feature_database, or mismatch → compute descriptors now
        if self.backbone is None:
            raise RuntimeError(
                "build_db_index requires either pre_navigation() (to build feature_database) "
                "or self.backbone to be set so descriptors can be computed."
            )

        logging.info("Building DB index by computing ResNet descriptors (no feature_database available)...")
        for i, fname in enumerate(files):
            path = os.path.join(self.DB_DIR, fname)
            img = Image.open(path).convert("RGB")
            desc = self.compute_descriptor(self.backbone, img)  # L2-normalized 512-d
            db_index.append((fname, desc.astype(np.float32)))

        return db_index

    def act(self):
        if self._state is None:
            return Action.IDLE

        phase = self._state[1]

        if phase == Phase.EXPLORATION:
            return Action.IDLE

        if phase == Phase.NAVIGATION:
            if self.nav_start_time is None:
                self.nav_start_time = time.time()
            if self.goal_id is None:
                self.goal_id = self.find_goal()
                if self.goal_id is None:
                    logging.error("Failed to find goal")
                    return Action.IDLE

        # NEW: precompute shortest path once
            if self.current_path is None:
                self.current_path = self.dijkstra_shortest_path(
                    self.current_id, self.goal_id
                )
                if self.current_path is None:
                    logging.error("Goal unreachable by Dijkstra; falling back")
                else:
                    logging.info(
                        f"Dijkstra path length: {len(self.current_path)} nodes "
                        f"(from {self.current_id} to {self.goal_id})"
                    )

            return self.select_action()
    
    # resnet model
    def build_backbone(self):
        model = models.resnet18(pretrained=True)
        backbone = nn.Sequential(*list(model.children())[:-1]) 
        backbone.eval()
        backbone.to(self.DEVICE)
        return backbone
    
    def compute_features(self):
        files = natsorted([x for x in os.listdir(self.save_dir) if x.endswith('.jpg')])
        self.num_images = len(files)
        self.resnet_descriptors = []

        logging.info(f"Computing ResNet features for {self.num_images} images...")
        for img_file in tqdm(files, desc="Extracting ResNet descriptors"):
            img = cv2.imread(os.path.join(self.save_dir, img_file))
            self.exploration_images.append(img)
            des = self.compute_descriptor(self.backbone, img)   # expect (512,) or (1,512)
            if des is not None:
                des = np.asarray(des).reshape(1, -1)            # (1,512)
                self.resnet_descriptors.append(des)             # append as row

        # stack into [num_images, 512]
        features = np.vstack(self.resnet_descriptors)           # (N,512)
        return features
    def codebook_init(self):
        if os.path.exists("resnet_descriptors.npy"):
            self.resnet_descriptors = np.load("resnet_descriptors.npy")
        if os.path.exists("codebook.pkl"):
            with open("codebook.pkl", "rb") as f:
                self.codebook = pickle.load(f)
    def see(self, fpv):
        """Receive first-person view"""
        if fpv is not None and len(fpv.shape) == 3:
            self.fpv = fpv.copy()
            self.fpv_fmap = self._extract_feature_map(self.fpv)
    def pre_navigation(self):
        """
        Pre-navigation setup:
        - Load exploration images from self.save_dir
        - Build (or load) a ResNet feature database
        - NO KMeans, NO VLAD, NO BallTree
        """
        files = natsorted([x for x in os.listdir(self.save_dir)
                           if x.endswith('.jpg')])
        self.num_images = len(files)

        self.exploration_images = []
        logging.info("Loading exploration images...")
        for img_file in tqdm(files, desc="Loading images"):
            img = cv2.imread(os.path.join(self.save_dir, img_file))
            self.exploration_images.append(img)

        # --- Build / load ResNet descriptors (no VLAD) ---
        feat_cache_path = "resnet_descriptors.npy"

        if self.resnet_descriptors is None:
            if os.path.exists(feat_cache_path):
                logging.info("Loading ResNet descriptor cache...")
                self.resnet_descriptors = np.load(feat_cache_path)
            else:
                logging.info("Computing ResNet features for database...")
                # Uses self.backbone and self.save_dir internally
                self.resnet_descriptors = self.compute_features()
                np.save(feat_cache_path, self.resnet_descriptors)

        # If cache shape doesn’t match number of images, recompute
        if self.resnet_descriptors.shape[0] != self.num_images:
            logging.info("ResNet descriptor cache size mismatch. Recomputing...")
            self.resnet_descriptors = self.compute_features()
            np.save(feat_cache_path, self.resnet_descriptors)

        # L2-normalize descriptors for cosine similarity
        feats = self.resnet_descriptors.astype(np.float32)
        norms = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-10
        self.feature_database = feats / norms   # [N, 512], each row unit-length

        logging.info(f"Feature database ready: {self.feature_database.shape[0]} images")

        # If you already added Dijkstra earlier, you can still do:
        # self.build_topology_graph()

    def plan_recovery_sequence(self):
        """
        SIMPLE stuck recovery:
        1. Back up 20 steps
        2. Turn 180 degrees
        3. Go forward 10 steps
        """
        self.recovery_queue.clear()
        self.in_recovery_mode = True

        # Simple: back up, turn around, go forward
        N_BACK = 20
        N_TURN = 18  # 180 degrees
        N_FWD = 10

        self.recovery_queue.extend([Action.BACKWARD] * N_BACK)
        self.recovery_queue.extend([Action.RIGHT] * N_TURN)
        self.recovery_queue.extend([Action.FORWARD] * N_FWD)

        logging.warning(f"STUCK! Executing turnaround: back {N_BACK}, turn 180°, forward {N_FWD}")
    
    def show_navigation_visualization(self, current_img, target_img, action, current_id, target_id):
        current_resized = cv2.resize(current_img, (320, 240))
        target_resized = cv2.resize(target_img, (320, 240))

        current_labeled = current_resized.copy()
        target_labeled = target_resized.copy()

        cv2.putText(current_labeled, f"Current View (ID: {current_id})", (10, 30),
                    cv2.FONT_ITALIC, 0.6, (0, 255, 0), 2)
        cv2.putText(current_labeled, f"Goal: {self.goal_id}", (10, 60),
                    cv2.FONT_ITALIC, 0.6, (255, 255, 0), 2)

        action_text = str(action).split('.')[-1]
        action_color = (255, 255, 255) if action_text == 'FORWARD' else (255, 165, 255)
        cv2.putText(current_labeled, f"Action: {action_text}", (10, 90),
                    cv2.FONT_ITALIC, 0.6, action_color, 2)
        cv2.putText(target_labeled, f"Target (ID: {target_id})", (10, 30),
                    cv2.FONT_ITALIC, 0.6, (0, 255, 0), 2)

        dist_to_goal = abs(self.goal_id - current_id)
        cv2.putText(target_labeled, f"Distance: {dist_to_goal} images", (10, 60),
                    cv2.FONT_ITALIC, 0.6, (255, 255, 255), 2)
        if self.goal_id != current_id:
            progress = 1.0 - (dist_to_goal / abs(
                self.goal_id - self.position_history[0] if len(self.position_history) > 0 else current_id))
            progress = max(0, min(1, progress))
        else:
            progress = 1.0

        bar_width = 300
        bar_height = 20
        cv2.rectangle(target_labeled, (10, 200), (10 + bar_width, 200 + bar_height),
                      (100, 100, 100), -1)
        cv2.rectangle(target_labeled, (10, 200), (10 + int(bar_width * progress), 200 + bar_height),
                      (0, 255, 0), -1)
        cv2.putText(target_labeled, f"{int(progress * 100)}%", (130, 215),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        visualization = np.hstack([current_labeled, target_labeled])

        # Display
        cv2.imshow('Navigation Visualization', visualization)
        cv2.waitKey(1)

    # All other methods from original file go here...
    # (I'm including the key select_action method with improvements)
    def get_neighbor(self, img, k=1):
        """
        Nearest neighbor search using direct ResNet descriptors
        and cosine similarity (no VLAD, no KMeans, no BallTree).

        Returns:
            indices:  list of length k with image indices
            distances: list of length k with "distances" = 1 - cosine_similarity
                       (so smaller is better, consistent with old code).
        """
        # Make sure feature database exists
        if self.feature_database is None:
            raise RuntimeError(
                "feature_database is None. Did you call pre_navigation() before navigation?"
            )

        # Query descriptor (already L2-normalized by compute_descriptor)
        q = self.compute_descriptor(self.backbone, img)  # shape (512,)
        q = np.asarray(q, dtype=np.float32).reshape(1, -1)

        # L2 normalize query to be safe
        q_norm = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-10)  # [1,512]

        # Cosine similarity = dot(q, db^T) because both are unit vectors
        # feature_database: [N, 512]
        sims = self.feature_database @ q_norm.T    # [N, 1]
        sims = sims.reshape(-1)                    # [N]

        # Higher sim = better → convert to "distance" so smaller is better
        # for compatibility with existing code.
        distances = 1.0 - sims                    # [N], in [0, 2] theoretically

        # Get top-k smallest distances
        idx_sorted = np.argsort(distances)[:k]
        top_dists = distances[idx_sorted]

        indices = idx_sorted.tolist()
        distances_list = top_dists.tolist()
        return indices, distances_list

    
    def select_action(self):
        """Enhanced action selection with wall awareness"""
        if self.fpv is None:
            return Action.IDLE

        # Execute recovery queue first
        if self.recovery_queue:
            action = self.recovery_queue.popleft()
            # Don't add recovery actions to history to avoid triggering "too many turns"
            # self.action_history.append(action)  # DISABLED

            # Exit recovery mode when queue is empty
            if len(self.recovery_queue) == 0:
                self.in_recovery_mode = False
                logging.info("Recovery complete, resuming normal navigation")

            return action

        # Mark recovery as complete if we get here
        self.in_recovery_mode = False

        # Decay cooldown
        if self.stuck_cooldown > 0:
            self.stuck_cooldown -= 1

        # Get current position via neighbor search
        indices, distances = self.get_neighbor(self.fpv, k=1)
        self.current_id = indices[0]
        confidence = distances[0]
        self.position_history.append(self.current_id)
        self.last_positions.append(self.current_id)

        # NEW: Track visits to current position
        visit_count = self.visited_cells.get(self.current_id, 0)
        self.visited_cells[self.current_id] = visit_count + 1

        # Decay old visit counts (keep only recent history)
        if len(self.visited_cells) > self.recent_visit_window:
            # Remove least recently visited cells
            sorted_cells = sorted(self.visited_cells.items(), key=lambda x: x[1])
            cells_to_remove = sorted_cells[:len(self.visited_cells) - self.recent_visit_window]
            for cell_id, _ in cells_to_remove:
                del self.visited_cells[cell_id]

        dist_to_goal = abs(self.goal_id - self.current_id)
        if len(self.position_history) % 25 == 0:
            total_visits = sum(self.visited_cells.values())
            unique_cells = len(self.visited_cells)
            logging.info(
                f"Position: {self.current_id} (visits: {visit_count + 1}), Goal: {self.goal_id}, "
                f"Distance: {dist_to_goal}, Confidence: {confidence:.4f}, "
                f"Explored: {unique_cells} cells, Total visits: {total_visits}"
            )

        # Goal check
        if self.nav_start_time is not None:
            elapsed = time.time() - self.nav_start_time
            tol = 2 if elapsed < 40 else 5 if elapsed < 80 else 10
            if abs(self.current_id - self.goal_id) <= tol:
                logging.info(f"Goal approx. reached (tol={tol}), checking in...")
                return Action.CHECKIN


        # Determine target frame
        # Determine target frame using Dijkstra path planning
        # (graph over exploration images, edges between neighbors)
        target_id = self.get_next_target_id_from_dijkstra(self.current_id, self.goal_id)

        # Clamp just in case
        target_id = max(0, min(target_id, self.num_images - 1))

        if target_id < len(self.exploration_images):
            target_img = self.exploration_images[target_id]
        else:
            # Fallback: safest default
            return Action.FORWARD


        # NEW: Check for stuck BEFORE trying to move
        # NEW: Check for stuck BEFORE trying to move
        if self.stuck_cooldown == 0 and self.is_stuck(dist_to_goal):
            logging.warning("Stuck detected! Initiating recovery...")
            self.consecutive_forward = 0
            self.stuck_cooldown = 40  # MUCH LONGER cooldown (was 15)
            self.position_history.clear()
            self.last_positions.clear()
            self.plan_recovery_sequence()

            if self.recovery_queue:
                action = self.recovery_queue.popleft()
                self.action_history.append(action)
                return action

        # NEW: Simple dead-end detection - if hitting wall, just turn around
        is_clear, forward_dist = self.is_path_clear(self.fpv)
        offset, left_free, right_free = self.corridor_center_offset(self.fpv)

        # Dead-end: both sides close AND forward blocked
        is_dead_end = (left_free < 70 and right_free < 70 and forward_dist < 40)
        if is_dead_end:
            # We hit a clear dead-end: mark this node as blocked for the
            # global planner so future Dijkstra runs will try to avoid it.
            logging.warning(
                f"DEAD END detected! L={left_free:.0f}, "
                f"R={right_free:.0f}, F={forward_dist:.0f}"
            )

            if self.current_id is not None and self.current_id != self.goal_id:
                self.blocked_nodes.add(self.current_id)
                # Invalidate any previously computed global path, since it may
                # have routed us straight into this dead-end.
                self.current_path = None
                self.path_valid_for_goal = None
                self.planned_path = None

            self.recovery_queue.clear()
            self.recovery_queue.extend([Action.BACKWARD] * 20)  # back up
            self.recovery_queue.extend([Action.RIGHT] * 18)     # turn 180°
            self.recovery_queue.extend([Action.FORWARD] * 10)   # go forward

            action = self.recovery_queue.popleft()
            self.consecutive_forward = 0
            self.consecutive_turns = 0
            return action

        # Close to a wall but not a full dead-end: do a mini recovery
        # This handles the "staring at a wall" situation.
    # Close to wall on one side - just steer away gently
      # Close to wall on one side - just steer away gently
        if forward_dist < 20 and self.consecutive_forward > 3:
            logging.info(f"Close to wall (dist={forward_dist}), small recovery turn")
            self.recovery_queue.clear()
            # tiny back-up, then turn toward free side
            self.recovery_queue.extend([Action.BACKWARD] * 3)
            turn_action = Action.RIGHT if right_free > left_free else Action.LEFT
            self.recovery_queue.extend([turn_action] * 6)
            action = self.recovery_queue.popleft()
            self.consecutive_forward = 0
            return action

        # NEW: Limit excessive forward movement (INCREASED LIMIT)
        MAX_CONSECUTIVE_FORWARD = 40  # DOUBLED from 20
        if self.consecutive_forward >= MAX_CONSECUTIVE_FORWARD:
            logging.info(f"Max forward limit ({MAX_CONSECUTIVE_FORWARD}) reached, reassessing")
            offset, left_free, right_free = self.corridor_center_offset(self.fpv)
            # Small turn to reassess the situation
            turn_action = Action.RIGHT if right_free > left_free else Action.LEFT
            self.recovery_queue.extend([turn_action] * 2)  # Just 2 turns
            self.consecutive_forward = 0
            action = self.recovery_queue.popleft()
            self.action_history.append(action)
            return action

        # Normal steering with visual turn estimation
        turn = self.compute_turn_direction(self.fpv, target_img)

        # NEW: Apply corridor centering bias
        offset, left_free, right_free = self.corridor_center_offset(self.fpv)

        # If we're significantly off-center, bias the turn decision
        if abs(offset) > 0.2:  # Significant offset
            if offset > 0.2 and turn != -1:  # Too close to left, bias right
                turn = 1
            elif offset < -0.2 and turn != 1:  # Too close to right, bias left
                turn = -1

        # NEW: Visit-aware navigation - avoid heavily visited areas
        if visit_count >= self.visit_penalty_threshold:
            logging.info(f"Heavily visited cell (visits={visit_count}), biasing toward exploration)")

            # Instead of using 'direction', we bias the turn based on corridor width
            # and relative wall distances (left_free, right_free)

            # If blocked or narrow corridor, pick the side with more space
            if abs(left_free - right_free) > 10:
                turn = 1 if right_free > left_free else -1
                logging.info("Choosing turn based on corridor asymmetry (avoid visited cell)")
            else:
                # Otherwise choose a deterministic but pseudo-random direction
                # based on visit count to avoid oscillation
                turn = 1 if (visit_count % 2 == 0) else -1
                logging.info("Using pseudo-random turn to escape visited cell")

            # We do NOT use a forward_id or direction anymore—Dijkstra handles global planning.
                # EXTREME over-visit: declare this node bad for global planning
        severe_overvisit_threshold = max(self.visit_penalty_threshold * 3, 50)

        if visit_count >= severe_overvisit_threshold:
            if self.current_id is not None and self.current_id != self.goal_id:
                if self.current_id not in self.blocked_nodes:
                    logging.warning(
                        f"Node {self.current_id} visited {visit_count} times – "
                        "marking as blocked for Dijkstra and replanning."
                    )
                    self.blocked_nodes.add(self.current_id)

                    # Invalidate current global path; next call to
                    # get_next_target_id_from_dijkstra will recompute.
                    self.current_path = None
                    self.path_valid_for_goal = None
                    self.planned_path = None


        # NEW: When no progress for a while, prefer FORWARD to break jittering
        if self.no_progress_steps > 10 and turn != 0:
            # Force forward occasionally to break oscillation
            if self.no_progress_steps % 5 == 0:
                logging.info("No progress - forcing forward to explore")
                turn = 0

        if turn == 0:
            action = Action.FORWARD
            self.consecutive_forward += 1
            self.consecutive_turns = 0  # Reset turn counter
        elif turn == 1:
            action = Action.RIGHT
            self.consecutive_forward = 0
            self.consecutive_turns += 1
        else:
            action = Action.LEFT
            self.consecutive_forward = 0
            self.consecutive_turns += 1

        # NEW: Detect jittering (many consecutive turns without forward)
        if self.consecutive_turns >= 8:
            logging.warning(f"Jittering detected ({self.consecutive_turns} turns), forcing forward burst")
            self.recovery_queue.clear()
            self.recovery_queue.extend([Action.FORWARD] * 10)
            self.consecutive_turns = 0
            action = self.recovery_queue.popleft()

        # Safety: avoid endless turning (DISABLED - causes oscillation with recovery)
        # Instead rely on stuck detection and forward limit
        # if self.consecutive_forward == 0 and len(self.action_history) >= 6:
        #     recent_actions = list(self.action_history)[-6:]
        #     if sum(1 for a in recent_actions if a in [Action.LEFT, Action.RIGHT]) >= 5:
        #         logging.info("Too many turns, forcing forward burst")
        #         self.recovery_queue.clear()
        #         self.recovery_queue.extend([Action.FORWARD] * 3)
        #         action = self.recovery_queue.popleft()

        self.action_history.append(action)

        if self.show_visualization:
            self.show_navigation_visualization(self.fpv, target_img, action,
                                            self.current_id, target_id)

        return action
    def reset(self):
        self.fpv = None
        self.goal_id = None
        self.current_id = None

        self.position_history.clear()
        self.action_history.clear()
        self.target_history = []

        self.consecutive_forward = 0
        self.consecutive_turns = 0

        self.stuck_cooldown = 0
        self.best_dist_to_goal = None
        self.no_progress_steps = 0

        self.recovery_queue.clear()
        self.visited_cells.clear()

        self.last_action = None

    @torch.no_grad()
    def compute_descriptor(self, backbone, img_):
        to_tensor_transform = transforms.ToTensor()
        if(isinstance(img_,torch.Tensor) == False):
            if(isinstance(img_, np.ndarray) == True):
                pil_image = Image.fromarray(img_)
                img_ = preprocess(pil_image)
                x = img_.to(self.DEVICE)
            else:
                img_ = preprocess(img_)
                x = img_.to(self.DEVICE)
        else:
            img_ = preprocess(img_).to(self.DEVICE)
        x = x.unsqueeze(0)
        feat = backbone(x)                           
        feat = feat.view(1, -1)                      
        # L2 normalize
        feat = feat / (feat.norm(p=2, dim=1, keepdim=True) + 1e-10)
        return feat.squeeze(0).cpu().numpy()   
    
    def compute_turn_direction(self, current_img, target_img,
                               #fmap1=None,
                               ratio_thresh=0.75,
                               min_matches=10,
                               pixel_threshold=10.0):
        """
        current_img, target_img: OpenCV BGR uint8 images.
        Returns:
            1  = turn one way (e.g. "right")
           -1  = turn the other way (e.g. "left")
            0  = no clear turn
        """
        #if fmap1 is None:
        #    fmap1 = self._extract_feature_map(current_img)
        # 1) Feature maps
        fmap1 = self._extract_feature_map(current_img)  # [C, Hf, Wf]
        fmap2 = self._extract_feature_map(target_img)   # [C, Hf, Wf]

        C, Hf, Wf = fmap1.shape

        # 2) Flatten to [N, C] and L2-normalize along C
        # N = Hf * Wf
        f1 = fmap1.view(C, -1).T    # [N, C]
        f2 = fmap2.view(C, -1).T    # [N, C]

        f1 = F.normalize(f1, p=2, dim=1)
        f2 = F.normalize(f2, p=2, dim=1)

        # 3) Similarity matrix: [N1, N2] = [N, N]
        # sim[i, j] = cosine similarity between feature i in img1 and feature j in img2
        # (matrix multiply because both are normalized)
        sim = torch.matmul(f1, f2.T)  # [N, N]

        # 4) KNN in descriptor space (k=2) + ratio test
        # We want, for each i in img1, the top-2 matches in img2
        top2_vals, top2_idx = sim.topk(k=2, dim=1)  # each row: best, second-best

        good_matches = []
        N = Hf * Wf

        for i in range(N):
            best_val = top2_vals[i, 0].item()
            second_val = top2_vals[i, 1].item()

            # Cosine similarity is in [-1,1]. Use ratio test on *distances* or *1 - sim*.
            # Simpler here: require best significantly better than second-best.
            # Approximate Lowe ratio in similarity domain:
            if best_val <= 0:      # reject obviously bad matches
                continue

            # Convert similarities to "distances" for ratio test:
            d1 = 1.0 - best_val
            d2 = 1.0 - second_val

            if d1 < ratio_thresh * d2:
                j = top2_idx[i, 0].item()
                good_matches.append((i, j))

        if len(good_matches) < min_matches:
            return 0

        # 5) Convert feature-map indices -> original pixel coordinates
        # For simplicity, map linearly: x_px ≈ (w + 0.5) * (W_img / Wf)
        H_img1, W_img1 = current_img.shape[:2]
        H_img2, W_img2 = target_img.shape[:2]

        stride_x1 = float(W_img1) / float(Wf)
        stride_x2 = float(W_img2) / float(Wf)

        x_displacements = []

        for i, j in good_matches:
            h1, w1 = divmod(i, Wf)
            h2, w2 = divmod(j, Wf)

            # (Optionally) enforce similar vertical row to reduce garbage matches:
            # if abs(h1 - h2) > 1: continue

            x1 = (w1 + 0.5) * stride_x1
            x2 = (w2 + 0.5) * stride_x2

            displacement = x2 - x1   # same sign convention as your SIFT code
            x_displacements.append(displacement)

        if len(x_displacements) == 0:
            return 0

        median_displacement = float(np.median(x_displacements))

        if median_displacement > pixel_threshold:
            return 1
        elif median_displacement < -pixel_threshold:
            return -1
        else:
            return 0
    # Placeholder - copy your existing methods here
    def prepare_for_resnet(self, img_bgr):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img_rgb)
        x = self.preprocess_no_resize(pil).unsqueeze(0)
        return x

    def _extract_feature_map(self, bgr_img: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        x = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            fmap = self.backbone_spatial(x)
        return fmap.squeeze(0)

    def corridor_sides_free_space(self, img):
        H, W = img.shape[:2]
        roi = img[H//2:, :]
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(roi_gray, 50, 150)
        
        cols = np.arange(W)
        left_cols  = cols[:W//3]
        right_cols = cols[-W//3:]

        def avg_free_dist(cols_subset):
            dists = []
            for c in cols_subset:
                col = edges[:, c]
                col_rev = col[::-1]
                wall_idx = np.argmax(col_rev > 0)
                if col_rev[wall_idx] == 0:
                    d = len(col_rev)
                else:
                    d = wall_idx
                dists.append(d)
            if len(dists) == 0:
                return 0.0
            return float(np.mean(dists))

        left_free  = avg_free_dist(left_cols)
        right_free = avg_free_dist(right_cols)
        return left_free, right_free
    
    def retrieve_matches(self, backbone, db_index, img, top_k=10):
        #for i, fname in enumerate(img_path):
        #    path = os.path.join(self.DB_DIR, fname)
        #print(type(img))

        q_desc = self.compute_descriptor(backbone, img)
     
        sims = []
        for fname, d_desc in db_index:
            s = float(np.dot(q_desc, d_desc)) 
            dist =  np.linalg.norm(q_desc - d_desc)
            sims.append((s, dist, fname))

        sims.sort(reverse=True, key=lambda x: x[0]) 
        return sims[:top_k]
    def is_corridor_like(self, left_free, right_free, min_free=10):
        # min_free is in pixels (in the bottom-half ROI)
        return (left_free  > min_free) and (right_free > min_free)
    
    def corridor_center_offset(self, img, eps=1e-3):
        left_free, right_free = self.corridor_sides_free_space(img)
        total = left_free + right_free + eps
        offset = (right_free - left_free) / total
        return offset, left_free, right_free

    # Include all other methods from your original file...
    # (For brevity, I've shown the key improvements. Copy remaining methods from dp.py)
    def prepare_for_resnet(self,img_bgr):
        # img_bgr: numpy [240,320,3] uint8 from cv2

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img_rgb)

        x = self.preprocess_no_resize(pil).unsqueeze(0)   # [1,3,240,320]
        return x
    def visualize_results(self, query_path, matches, out_path):
        fig, axes = plt.subplots(1, len(matches) + 1, figsize=(4 * (len(matches) + 1), 4))

        q_img = Image.open(query_path).convert("RGB")
        axes[0].imshow(q_img)
        axes[0].set_title("Query")
        axes[0].axis("off")

        for i, (score, fname) in enumerate(matches, start=1):
            m_img = Image.open(os.path.join(self.DB_DIR, fname)).convert("RGB")
            axes[i].imshow(m_img)
            axes[i].set_title(f"{fname}\ncos sim = {score:.3f}", fontsize=8)
            axes[i].axis("off")

        plt.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)

# using resnet backbone to do image retrieval and compare query images with database images
# then using cosine similarity to find individuals
def main():
    import vis_nav_game
    model = Autonnomous_resnet_Navigator()
    model.backbone = model.build_backbone()
    model.pre_navigation()
    # If you want to reuse pre_navigation’s features for retrieval,
    # you can optionally call:
    # model.pre_navigation()
    # Then build_db_index() will reuse model.feature_database.
    start_id = 0
    model.db_index = model.build_db_index()
    goal_id = model.num_images - 1
    model.build_topology_graph()
    model.plan_full_path(start_id, goal_id)
    model.save_path_to_file("planned_path.txt")
    model.save_graph_with_path_image("nav_graph.png", start_id=start_id, goal_id=goal_id)
    print("model device:", next(model.backbone_spatial.parameters()).device)

    print("*" * 30)
    print("dvl2013 Auto Target Planner")
    print("*" * 30)
    vis_nav_game.play(the_player=model)

if __name__ == "__main__":
    main()    