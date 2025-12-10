import os
import time
import logging
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
from natsort import natsorted
import networkx as nx

from vis_nav_game import Player, Action, Phase


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


class AutonomousResnetPathFollower(Player):
    """
    Very simple, fast agent:

    - Offline:
        * Loads exploration images
        * Computes ResNet global descriptors
        * Builds a 1D chain graph over images
        * Runs Dijkstra once to get the global shortest path

    - Online (during NAVIGATION):
        * Each frame:
            - Localize current FPV -> nearest image index (ResNet descriptor)
            - Advance an index along the precomputed path
            - Use simple corridor geometry to move:
                + If path ahead clear and centered -> FORWARD
                + If close to a wall -> small turn away
                + If forward blocked -> turn toward more free space
        * When near the goal -> CHECKIN
    """

    def __init__(self, device="cuda"):
        super().__init__()

        # --- Device / model state ---
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.backbone = None  # global descriptor backbone

        # Data dirs
        self.DATA_ROOT = "./data"
        self.save_dir = os.path.join(self.DATA_ROOT, "images_subsample")
        if not os.path.exists(self.save_dir):
            raise FileNotFoundError(f"Directory {self.save_dir} does not exist")

        # History (for debugging / minor logic)
        from collections import deque  # if not already imported at top
        self.position_history = deque(maxlen=50)

        # Image DB / descriptors
        self.exploration_images = []
        self.num_images = 0
        self.feature_database = None  # [N, 512] L2-normalized ResNet descriptors
        # Global path / graph
        self.graph = None               # networkx graph
        self.planned_path = None        # list of node ids (image indices)
        self.path_index_map = {}        # node_id -> index along planned_path
        self.path_progress_idx = 0      # how far along planned_path we've "reached"

        # Navigation state
        self.fpv = None           # current first-person view (BGR)
        self.goal_id = None       # final node id (image index)
        self.current_id = None    # nearest image index to current FPV

        self.nav_start_time = None
        self.time_budget = 60.0   # seconds (for your own logging)

        # History (mainly for debugging / future tuning)
        self.position_history = deque(maxlen=50)

        # Simple preprocessing for ResNet global descriptors
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        logging.info("AutonomousResnetPathFollower initialized")

    # -------------------------------------------------------------------------
    # Model / descriptor utilities
    # -------------------------------------------------------------------------
    def build_backbone(self):
        """
        Build a ResNet-18 backbone for global descriptors (avgpool output).
        """
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        backbone = nn.Sequential(*list(resnet.children())[:-1])  # up to avgpool
        backbone.eval()
        backbone.to(self.device)
        logging.info("ResNet-18 backbone built and moved to device")
        return backbone

    @torch.no_grad()
    def compute_descriptor(self, img_bgr):
        """
        Compute a single 512-D global descriptor for a BGR uint8 image.
        """
        if img_bgr is None:
            raise ValueError("compute_descriptor: img_bgr is None")

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img_rgb)
        x = self.preprocess(pil).unsqueeze(0).to(self.device)  # [1,3,224,224]

        feat = self.backbone(x)          # [1,512,1,1]
        feat = feat.view(1, -1)          # [1,512]
        feat = feat / (feat.norm(p=2, dim=1, keepdim=True) + 1e-10)
        return feat.squeeze(0).cpu().numpy()  # [512]

    def pre_navigation(self):
        """
        Offline pre-navigation:

        - Load exploration images from self.save_dir
        - Compute or load ResNet descriptors
        - Build L2-normalized feature database [N,512]
        """
        files = natsorted([f for f in os.listdir(self.save_dir) if f.lower().endswith(".jpg")])
        self.num_images = len(files)
        if self.num_images == 0:
            raise RuntimeError(f"No .jpg images found in {self.save_dir}")

        logging.info(f"Loading {self.num_images} exploration images...")
        self.exploration_images = []
        for img_file in tqdm(files, desc="Loading images"):
            img = cv2.imread(os.path.join(self.save_dir, img_file))
            if img is None:
                raise RuntimeError(f"Failed to load image {img_file}")
            self.exploration_images.append(img)

        # Cache path for descriptors
        feat_cache_path = "resnet_descriptors_fast.npy"

        if os.path.exists(feat_cache_path):
            logging.info("Loading cached ResNet descriptors...")
            feats = np.load(feat_cache_path)
            if feats.shape[0] != self.num_images:
                logging.warning(
                    "Cached descriptor count mismatch, recomputing descriptors..."
                )
                feats = self._compute_all_descriptors()
                np.save(feat_cache_path, feats)
        else:
            logging.info("Computing ResNet descriptors for database...")
            feats = self._compute_all_descriptors()
            np.save(feat_cache_path, feats)

        # L2 normalize
        feats = feats.astype(np.float32)
        norms = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-10
        self.feature_database = feats / norms
        logging.info(f"Feature database ready: {self.feature_database.shape}")

    def _compute_all_descriptors(self):
        """
        Helper: compute descriptors for all exploration images.
        """
        descs = []
        for img in tqdm(self.exploration_images, desc="Extracting descriptors"):
            d = self.compute_descriptor(img)
            descs.append(d.reshape(1, -1))
        feats = np.vstack(descs)  # [N,512]
        return feats

    def get_neighbor(self, img_bgr, k=1):
        """
        Nearest neighbor search in descriptor space.

        Returns:
            indices: list of top-k indices
            distances: corresponding "distances" = 1 - cosine_similarity
        """
        if self.feature_database is None:
            raise RuntimeError("feature_database is None. Call pre_navigation() first.")

        q_desc = self.compute_descriptor(img_bgr).astype(np.float32).reshape(1, -1)
        q_norm = q_desc / (np.linalg.norm(q_desc, axis=1, keepdims=True) + 1e-10)

        sims = self.feature_database @ q_norm.T  # [N,1]
        sims = sims.reshape(-1)                  # [N]
        distances = 1.0 - sims                  # "distance" (smaller is better)

        idx_sorted = np.argsort(distances)[:k]
        top_dists = distances[idx_sorted]

        return idx_sorted.tolist(), top_dists.tolist()

    # -------------------------------------------------------------------------
    # Graph + path planning
    # -------------------------------------------------------------------------
    def build_topology_graph(self):
        """
        Builds a simple chain graph: 0 -- 1 -- 2 -- ... -- (N-1)

        This matches your earlier assumption that image indices follow the maze.
        """
        G = nx.Graph()
        N = self.num_images
        for i in range(N):
            G.add_node(i)
        for i in range(N - 1):
            G.add_edge(i, i + 1, weight=1.0)
        self.graph = G
        logging.info(f"Built chain graph with {G.number_of_nodes()} nodes and "
                     f"{G.number_of_edges()} edges.")

    def plan_full_path(self, start_id: int, goal_id: int):
        """
        Use NetworkX Dijkstra to compute the full shortest path.
        """
        if self.graph is None:
            raise RuntimeError("Graph not built. Call build_topology_graph() first.")

        if start_id not in self.graph or goal_id not in self.graph:
            raise ValueError(f"Invalid start/goal ids: {start_id}, {goal_id}")

        path = nx.dijkstra_path(self.graph, source=start_id, target=goal_id, weight="weight")
        self.planned_path = path
        self.path_index_map = {node: idx for idx, node in enumerate(path)}
        self.path_progress_idx = 0

        logging.info(
            f"Planned global path from {start_id} to {goal_id}, length={len(path)} nodes."
        )

    # -------------------------------------------------------------------------
    # Perception: corridor + free space
    # -------------------------------------------------------------------------
    def is_path_clear(self, img, min_clear_distance=35):
        """
        Check if there's enough clear space ahead in the center-bottom region.

        Returns:
            (is_clear: bool, forward_distance: float)
        """
        H, W = img.shape[:2]
        roi = img[H // 2:, W // 3: 2 * W // 3]  # center third, bottom half
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(roi_gray, 50, 150)

        center_cols = edges[:, edges.shape[1] // 3: 2 * edges.shape[1] // 3]

        forward_clear = 0
        for row_idx in range(center_cols.shape[0] - 1, -1, -1):
            if np.any(center_cols[row_idx, :] > 0):
                forward_clear = center_cols.shape[0] - row_idx
                break
        else:
            forward_clear = center_cols.shape[0]

        is_clear = forward_clear > min_clear_distance
        return is_clear, float(forward_clear)

    def corridor_sides_free_space(self, img):
        """
        Measure average free distance (in pixels) on left/right bottom-half of image.
        """
        H, W = img.shape[:2]
        roi = img[H // 2:, :]
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(roi_gray, 50, 150)

        cols = np.arange(W)
        left_cols = cols[: W // 3]
        right_cols = cols[-W // 3:]

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
            if not dists:
                return 0.0
            return float(np.mean(dists))

        left_free = avg_free_dist(left_cols)
        right_free = avg_free_dist(right_cols)
        return left_free, right_free

    def corridor_center_offset(self, img, eps=1e-3):
        """
        Returns:
            offset in [-1,1], left_free, right_free

            offset > 0  => more free space on RIGHT side (you are closer to left wall)
            offset < 0  => more free space on LEFT side (you are closer to right wall)
        """
        left_free, right_free = self.corridor_sides_free_space(img)
        total = left_free + right_free + eps
        offset = (right_free - left_free) / total
        return offset, left_free, right_free

    # -------------------------------------------------------------------------
    # Game integration
    # -------------------------------------------------------------------------
    def see(self, fpv):
        """
        Called by the game: receives first-person view (BGR).
        """
        if fpv is not None and len(fpv.shape) == 3:
            self.fpv = fpv.copy()

    def act(self):
        """
        Main control entry point called by vis_nav_game each step.
        """
        if self._state is None:
            return Action.IDLE

        phase = self._state[1]

        if phase == Phase.EXPLORATION:
            # This agent only handles navigation; exploration handled elsewhere.
            return Action.IDLE

        if phase == Phase.NAVIGATION:
            # Initialize timing once
            if self.nav_start_time is None:
                self.nav_start_time = time.time()

            # Ensure goal and global path are set
            if self.goal_id is None:
                if self.planned_path is None:
                    logging.error("No planned_path set before NAVIGATION.")
                    return Action.IDLE
                self.goal_id = self.planned_path[-1]

            # Fast path-following controller
            return self.select_action_fast()

        return Action.IDLE

    def select_action_fast(self):
        """
        Lightweight, preplanned path follower:

        1. Localize FPV -> nearest image index (ResNet descriptor)
        2. Advance along `self.planned_path` (monotonic index)
        3. Use simple corridor geometry for control:
            - If forward blocked -> turn toward more free side
            - Else if off-center -> small turn to re-center
            - Else -> FORWARD
        4. When near goal -> CHECKIN
        """
        if self.fpv is None:
            return Action.IDLE

        # Localize current position via nearest neighbor
        indices, distances = self.get_neighbor(self.fpv, k=1)
        self.current_id = indices[0]
        self.position_history.append(self.current_id)

        dist_to_goal = abs(self.goal_id - self.current_id)

        # Update progress index along the precomputed path (monotonic)
        if self.current_id in self.path_index_map:
            idx_on_path = self.path_index_map[self.current_id]
            if idx_on_path > self.path_progress_idx:
                self.path_progress_idx = idx_on_path

        # If we're close enough to the end in terms of path index or id -> CHECKIN
        if self.path_progress_idx >= len(self.planned_path) - 3 or dist_to_goal <= 3:
            logging.info(
                f"Near goal: current_id={self.current_id}, "
                f"goal_id={self.goal_id}, path_idx={self.path_progress_idx}"
            )
            return Action.CHECKIN

        # Basic corridor / obstacle perception
        is_clear, forward_dist = self.is_path_clear(self.fpv, min_clear_distance=30)
        offset, left_free, right_free = self.corridor_center_offset(self.fpv)

        # 1) If forward is clearly blocked or very close, turn toward more free side
        if (not is_clear) or (forward_dist < 20):
            turn_action = Action.RIGHT if right_free > left_free else Action.LEFT
            logging.debug(
                f"Forward blocked (dist={forward_dist:.1f}), "
                f"left_free={left_free:.1f}, right_free={right_free:.1f}, "
                f"turn={turn_action}"
            )
            return turn_action

        # 2) If significantly off-center, steer gently back to the middle
        if abs(offset) > 0.2:
            # offset > 0 => closer to left wall -> turn RIGHT
            if offset > 0:
                return Action.RIGHT
            else:
                return Action.LEFT

    
        # 3) Otherwise, just go forward aggressively
        return Action.FORWARD

    def reset(self):
        self.fpv = None
        self.goal_id = None
        self.current_id = None
        self.nav_start_time = None

        # If the object didn’t have position_history for some reason,
        # recreate it; otherwise just clear.
        if not hasattr(self, "position_history"):
            from collections import deque
            self.position_history = deque(maxlen=50)
        else:
            self.position_history.clear()

        self.path_progress_idx = 0



def main():
    import vis_nav_game

    model = AutonomousResnetPathFollower()

    # Build backbone and pre-navigation DB
    model.backbone = model.build_backbone()
    model.pre_navigation()

    # Build graph and precompute global path
    start_id = 0
    goal_id = model.num_images - 1
    model.build_topology_graph()
    model.plan_full_path(start_id, goal_id)

    logging.info(f"Model device: {next(model.backbone.parameters()).device}")

    print("*" * 30)
    print("Preplanned Fast Path Follower")
    print("*" * 30)

    vis_nav_game.play(the_player=model)


if __name__ == "__main__":
    main()
