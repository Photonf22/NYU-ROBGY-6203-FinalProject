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
from sklearn.cluster import KMeans
from sklearn.neighbors import BallTree
from tqdm import tqdm
from natsort import natsorted
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
        self.save_dir = "data/images_subsample"
        if not os.path.exists(self.save_dir):
            raise FileNotFoundError(f"Directory {self.save_dir} does not exist")
        if os.path.exists("codebook.pkl"):
            with open("codebook.pkl", "rb") as f:
                self.codebook = pickle.load(f)
        self.DATA_ROOT = "./data/"
        #QUERY_DIR = os.path.join(DATA_ROOT, "query")
        self.DB_DIR = os.path.join(self.DATA_ROOT, "images_subsample")
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone = None
        self.fpv_fmap = None
        self.resnet_descriptors = None
        self.codebook = None
        self.database = None
        self.turn_action =  Action.RIGHT # default turn action
        self.exploration_images = []
        self.stuck_cooldown = 0          # steps until we allow another stuck trigger
        self.best_dist_to_goal = None    # best (smallest) distance so far
        self.no_progress_steps = 0       # steps since last improvement
        self.db_index= None
        self.fpv = None
        self.goal_id = None
        self.current_id = None
        self.num_images = 0
        self.nav_start_time = None
        self.time_budget = 60.0
        self.lazy_indexing = False
        self.position_history = deque(maxlen=20)
        self.action_history = deque(maxlen=5)
        self.consecutive_forward = 0
        self.recovery_queue = deque()
        self.vlad_cache_dir = os.path.join(os.getcwd(), "vlad_cache")
        os.makedirs(self.vlad_cache_dir, exist_ok=True)
        self._state = None
        self.show_visualization = True
        self.visualization_window = None
        self.target_comparison_window = None

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
    def prepare_for_resnet(self,img_bgr):
        # img_bgr: numpy [240,320,3] uint8 from cv2

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img_rgb)

        x = self.preprocess_no_resize(pil).unsqueeze(0)   # [1,3,240,320]
        return x

    def get_vlad(self, img):
        """
        Compute VLAD (Vector of Locally Aggregated Descriptors) descriptor for a given image
        """
        # We use a SIFT in combination with VLAD as a feature extractor as it offers several benefits
        # 1. SIFT features are invariant to scale and rotation changes in the image
        # 2. SIFT features are designed to capture local patterns which makes them more robust against noise
        # 3. VLAD aggregates local SIFT descriptors into a single compact representation for each image
        # 4. VLAD descriptors typically require less memory storage compared to storing the original set of SIFT
        # descriptors for each image. It is more practical for storing and retrieving large image databases efficicently.

        # Pass the image to sift detector and get keypoints + descriptions
        # Again we only need the descriptors
        #_, des = self.sift.detectAndCompute(img, None)
        #img = self.prepare_for_resnet(img).to(device)
        des = self.compute_descriptor(self.backbone, img)  # likely a torch.Tensor

        # Ensure: 2D, N x 512
        if isinstance(des, torch.Tensor):
            if des.ndim == 1:         # [512] -> [1,512]
                des = des.unsqueeze(0)
            des_np = des.detach().cpu().numpy().astype(np.float32, copy=False)
        else:
            des_np = np.asarray(des, dtype=np.float32)
            if des_np.ndim == 1:      # (512,) -> (1,512)
                des_np = des_np.reshape(1, -1)
        des = des_np
        #print("des_np shape for KMeans:", des.shape)  # should be (N, 512)
        # We then predict the cluster labels using the pre-trained codebook
        # Each descriptor is assigned to a cluster, and the predicted cluster label is returned
        pred_labels = self.codebook.predict(des)
        # Get number of clusters that each descriptor belongs to
        centroids = self.codebook.cluster_centers_
        # Get the number of clusters from the codebook
        k = self.codebook.n_clusters
        VLAD_feature = np.zeros([k, des.shape[1]])

        # Loop over the clusters
        for i in range(k):
            # If the current cluster label matches the predicted one
            if np.sum(pred_labels == i) > 0:
                # Then, sum the residual vectors (difference between descriptors and cluster centroids)
                # for all the descriptors assigned to that clusters
                # axis=0 indicates summing along the rows (each row represents a descriptor)
                # This way we compute the VLAD vector for the current cluster i
                # This operation captures not only the presence of features but also their spatial distribution within the image
                VLAD_feature[i] = np.sum(des[pred_labels==i, :] - centroids[i], axis=0)
        VLAD_feature = VLAD_feature.flatten()
        # Apply power normalization to the VLAD feature vector
        # It takes the element-wise square root of the absolute values of the VLAD feature vector and then multiplies 
        # it by the element-wise sign of the VLAD feature vector
        # This makes the resulting descriptor robust to noice and variations in illumination which helps improve the 
        # robustness of VPR systems
        VLAD_feature = np.sign(VLAD_feature)*np.sqrt(np.abs(VLAD_feature))
        # Finally, the VLAD feature vector is normalized by dividing it by its L2 norm, ensuring that it has unit length
        VLAD_feature = VLAD_feature/np.linalg.norm(VLAD_feature)

        return VLAD_feature
    def _extract_feature_map(self, bgr_img: np.ndarray) -> torch.Tensor:
        """
        bgr_img: OpenCV BGR uint8 image, shape [H, W, 3]
        returns: feature map [C, Hf, Wf] on self.device
        """
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        x = self.preprocess(pil_img).unsqueeze(0).to(self.device)  # [1,3,224,224]
        with torch.no_grad():
            fmap = self.backbone_spatial(x)   # [1, C, Hf, Wf]
        return fmap[0]                        # [C, Hf, Wf]
    
    def codebook_init(self):
        if os.path.exists("resnet_descriptors.npy"):
            self.resnet_descriptors = np.load("resnet_descriptors.npy")
        if os.path.exists("codebook.pkl"):
            with open("codebook.pkl", "rb") as f:
                self.codebook = pickle.load(f)

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

    def build_db_index(self, backbone):
        db_index = []  
        files = sorted([f for f in os.listdir(self.DB_DIR) if f.lower().endswith(".jpg")])

        for i, fname in enumerate(files):
            path = os.path.join(self.DB_DIR, fname)
            img = Image.open(path).convert("RGB")
            desc = self.compute_descriptor(backbone, img)
            db_index.append((fname, desc))
        return db_index
    def is_stuck(self, dist_to_goal):
        # Need some history, but not too much
        if len(self.position_history) < 10:
            return False

        # Simple cooldown so we don't trigger every frame
        if self.stuck_cooldown > 0:
            return False

        # Look at the last 10 positions
        recent = list(self.position_history)[-10:]
        unique = len(set(recent))

        from collections import Counter
        counts = Counter(recent)
        most_common_id, most_common_count = counts.most_common(1)[0]
        freq = most_common_count / len(recent)

        # More aggressive stuck criteria:
        # - we're bouncing among very few indices
        # - one index dominates the window
        stuck_pos = (unique <= 3 and freq > 0.6)

        # Progress logic: track best distance to goal so far
        if self.best_dist_to_goal is None or dist_to_goal < self.best_dist_to_goal - 2:
            self.best_dist_to_goal = dist_to_goal
            self.no_progress_steps = 0
        else:
            self.no_progress_steps += 1

        # Fewer steps allowed without improving
        stuck_progress = self.no_progress_steps > 20

        return stuck_pos and stuck_progress

    
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

    def reset(self):
        self.fpv = None
        self.goal_id = None
        self.current_id = None
        self.position_history.clear()
        self.action_history.clear()
        self.consecutive_forward = 0

        self.stuck_cooldown = 0
        self.best_dist_to_goal = None
        self.no_progress_steps = 0

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
    
    
    def get_neighbor(self, img, k=1):
        q_vlad = self.get_vlad(img)

        if not self.lazy_indexing:
            distances, indices = self.tree.query(q_vlad.reshape(1, -1), k)
            return indices[0], distances[0]
        best = []
        for idx, db_img in enumerate(self.exploration_images):
            cache_file = os.path.join(self.vlad_cache_dir, f"{idx}.npy")
            if os.path.exists(cache_file):
                v = np.load(cache_file)
            else:
                v = self.get_vlad(db_img)
                np.save(cache_file, v)

            d = np.linalg.norm(q_vlad - v)
            best.append((d, idx))
        best.sort(key=lambda x: x[0])
        topk = best[:k]
        indices = [i for _, i in topk]
        distances = [d for d, _ in topk]
        return indices, distances
    
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

    def corridor_sides_free_space(self, img):
        H, W = img.shape[:2]
        roi = img[H//2:, :]           # bottom half
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(roi_gray, 50, 150)    # tune thresholds

        # We'll measure free distance in columns in left and right thirds
        cols = np.arange(W)
        left_cols  = cols[:W//3]
        right_cols = cols[-W//3:]

        def avg_free_dist(cols_subset):
            dists = []
            for c in cols_subset:
                col = edges[:, c]             # bottom-half column, shape [H//2]
                # We search from bottom up: index 0 is top of ROI, so reverse
                col_rev = col[::-1]
                wall_idx = np.argmax(col_rev > 0)   # first edge pixel from bottom
                if col_rev[wall_idx] == 0:
                    # No edge in this column → treat as very far
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
    
    def corridor_center_offset(self, img, eps=1e-3):
        left_free, right_free = self.corridor_sides_free_space(img)
        total = left_free + right_free + eps
        # normalized asymmetry: >0 means closer to left wall (need to steer right)
        offset = (right_free - left_free) / total
        return offset, left_free, right_free

    def is_corridor_like(self, left_free, right_free, min_free=10):
        # min_free is in pixels (in the bottom-half ROI)
        return (left_free  > min_free) and (right_free > min_free)

    def plan_recovery_sequence(self):
        # Inspect current FPV to decide how aggressive we need to be
        offset, left_free, right_free = self.corridor_center_offset(self.fpv)

        # Basic recovery: back up until we see a corridor-like situation
        # Then center, then forward.
        self.recovery_queue.clear()

        # Phase 1: back up for a fixed number of steps to get away from the current wall
        N_BACK = 15
        self.recovery_queue.extend([Action.BACKWARD] * N_BACK)

        # Phase 2: rotate toward the more open side (offset sign tells you that)
        # offset > 0 → closer to left wall → turn right
        if offset > 0:
            turn_action = Action.RIGHT
        else:
            turn_action = Action.LEFT

        N_TURN = 20
        self.recovery_queue.extend([turn_action] * N_TURN)

        # Phase 3: go forward for a while
        N_FWD = 1
        self.recovery_queue.extend([Action.FORWARD] * N_FWD)

    def select_action(self):
        if self.fpv is None:
            return Action.IDLE

        if self.recovery_queue:
            action = self.recovery_queue.popleft()
            self.action_history.append(action)
            # no get_neighbor, no compute_turn_direction here
            return action
        # --- 1. Cooldown decay so stuck doesn't trigger continuously ---
        if self.stuck_cooldown > 0:
            self.stuck_cooldown -= 1

        # --- 2. FAST PATH: if we are in a scripted recovery, just execute it ---
        #     NO neighbor search, NO turn estimation here.
        if self.recovery_queue:
            # Optionally refine: if we're in FORWARD phase of recovery,
            # recompute offset and bias to stay centered.
            action = self.recovery_queue[0]

            if action == Action.FORWARD:
                offset, left_free, right_free = self.corridor_center_offset(self.fpv)
                if offset > 0.15:
                    action = Action.RIGHT   # adjust to re-center
                elif offset < -0.15:
                    action = Action.LEFT

            self.recovery_queue.popleft()
            return action

        # --- 3. Heavy neighbor search only when NOT in recovery ---
        indices, distances = self.get_neighbor(self.fpv, k=1)
        self.current_id = indices[0]
        confidence = distances[0]
        self.position_history.append(self.current_id)

        dist_to_goal = abs(self.goal_id - self.current_id)
        if len(self.position_history) % 25 == 0:
            logging.info(
                f"Position: {self.current_id}, Goal: {self.goal_id}, "
                f"Distance: {dist_to_goal}, Confidence: {confidence:.4f}"
            )

        # Goal check: close enough, check in
        if abs(self.current_id - self.goal_id) <= 2:
            logging.info("Goal reached! Checking in...")
            return Action.CHECKIN

        # --- 4. Choose a target frame ahead along the sequence ---
        direction = 1 if self.goal_id > self.current_id else -1
        distance_to_goal = abs(self.goal_id - self.current_id)
        if distance_to_goal > 30:
            lookahead = 7
        elif distance_to_goal > 10:
            lookahead = 5
        else:
            lookahead = 3

        target_id = self.current_id + direction * lookahead
        target_id = max(0, min(target_id, self.num_images - 1))

        if target_id < len(self.exploration_images):
            target_img = self.exploration_images[target_id]
        else:
            # Fallback if index is weird
            return Action.FORWARD

        if self.recovery_queue:
            action = self.recovery_queue.popleft()
            self.action_history.append(action)
            if self.show_visualization:
                self.show_navigation_visualization(self.fpv, target_img, action,
                                                   self.current_id, target_id)
            return action
        if self.is_stuck(dist_to_goal):
            logging.warning("Stuck detected! Recovery with corridor centering...")
            self.consecutive_forward = 0
            self.stuck_cooldown = 10
            self.position_history.clear()

            self.plan_recovery_sequence()  # uses corridor_center_offset(self.fpv)

            action = self.recovery_queue.popleft()
            self.action_history.append(action)
            return action



        # --- 6. Normal steering with visual turn estimation (heavy) ---
        turn = self.compute_turn_direction(self.fpv, target_img)
        if turn == 0:
            action = Action.FORWARD
            self.consecutive_forward += 1
        elif turn == 1:
            action = Action.RIGHT
            self.consecutive_forward = 0
        else:
            action = Action.LEFT
            self.consecutive_forward = 0

        # Optional safety: avoid endless turning in place
        if self.consecutive_forward == 0 and len(self.action_history) >= 4:
            recent_actions = list(self.action_history)[-4:]
            if all(a in [Action.LEFT, Action.RIGHT] for a in recent_actions):
                logging.info("Too many turns, forcing forward burst")
                # Force a short burst of forward motion
                self.recovery_queue.clear()
                self.recovery_queue.extend([Action.FORWARD] * 4)
                action = self.recovery_queue.popleft()
                self.action_history.append(action)
                return action

        self.action_history.append(action)

        if self.show_visualization:
            self.show_navigation_visualization(self.fpv, target_img, action,
                                            self.current_id, target_id)

        return action

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

            return self.select_action()

        return Action.IDLE
    def see(self, fpv):
        """Receive first-person view"""
        if fpv is not None and len(fpv.shape) == 3:
            self.fpv = fpv.copy()
            self.fpv_fmap = self._extract_feature_map(self.fpv)
    
    def pre_navigation(self):
        self.codebook_init()
        files = natsorted([x for x in os.listdir(self.save_dir) if x.endswith('.jpg')])
        self.num_images = len(files)
        logging.info("Loading exploration images...")
        for img_file in tqdm(files, desc="Loading images"):
            img = cv2.imread(os.path.join(self.save_dir, img_file))
            self.exploration_images.append(img)
        if self.codebook is None:
            if self.resnet_descriptors is None:
                logging.info("Computing resnet features for codebook...")
                self.resnet_descriptors = self.compute_features()
                np.save("self.resnet_descriptors.npy", self.resnet_descriptors)
            logging.info("Building codebook (K-means clustering)...")
            self.codebook = KMeans(
                n_clusters=64,
                init='k-means++',
                n_init=5,
                verbose=0
            ).fit(self.resnet_descriptors)
            
            with open("codebook.pkl", "wb") as f:
                pickle.dump(self.codebook, f)
        
        else:
            logging.info("Loaded codebook from cache")
        
        if self.lazy_indexing:
            self.database = None
            self.tree = None
            logging.info("Lazy indexing enabled: skipping full VLAD/BallTree build")
        else:
            vlad_cache_path = "vlad_database.npy"
            if os.path.exists(vlad_cache_path):
                cached = np.load(vlad_cache_path)
                if cached.shape[0] == self.num_images:
                    logging.info("Loaded VLAD database from cache")
                    self.database = cached
                else:
                    logging.info("VLAD cache size mismatch. Recomputing VLAD database...")
                    self.database = []
                    for img in tqdm(self.exploration_images, desc="Computing VLAD"):
                        vlad = self.get_vlad(img)
                        self.database.append(vlad)
                    self.database = np.asarray(self.database)
                    np.save(vlad_cache_path, self.database)
            else:
                logging.info("Computing VLAD database...")
                self.database = []
                for img in tqdm(self.exploration_images, desc="Computing VLAD"):
                    vlad = self.get_vlad(img)
                    self.database.append(vlad)
                self.database = np.asarray(self.database)
                np.save(vlad_cache_path, self.database)
            balltree_cache_path = "balltree.pkl"
            if os.path.exists(balltree_cache_path):
                try:
                    with open(balltree_cache_path, "rb") as f:
                        self.tree = pickle.load(f)
                    _ = self.tree.query(self.database[0].reshape(1, -1), k=1)
                    logging.info("Loaded BallTree from cache")
                except Exception:
                    logging.info("BallTree cache invalid. Rebuilding...")
                    self.tree = BallTree(self.database, leaf_size=64)
                    with open(balltree_cache_path, "wb") as f:
                        pickle.dump(self.tree, f)
            else:
                logging.info("Building BallTree for fast search...")
                self.tree = BallTree(self.database, leaf_size=64)
                with open(balltree_cache_path, "wb") as f:
                    pickle.dump(self.tree, f)
            logging.info(f"Ready! Database: {len(self.database)} images")

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
    model.db_index = model.build_db_index(model.backbone)
    print("model device:", next(model.backbone_spatial.parameters()).device)
    #query_files = sorted(
    #    f for f in os.listdir(QUERY_DIR)
    #    if f.lower().endswith(".jpg")
    #)

    #for qf in query_files:
    #    q_path = os.path.join(QUERY_DIR, qf)
    #    print(f"\nQuery of Images Provided: {qf} ")
    #    matches = retrieve_matches(backbone, db_index, q_path, top_k=TOP_K)

        #for rank, (score, fname) in enumerate(matches, start=1):
        #    print(f"#{rank}: {fname} (similarity = {score:.4f})")

        #vis_path = f"results_{os.path.splitext(qf)[0]}.png"
        #visualize_results(q_path, matches, vis_path)
        #print(f"Saved visualization to {vis_path}")
    print("*" * 30)
    print("dvl2013 Auto Target Planner")
    print("*" * 30)
    vis_nav_game.play(the_player=model)

if __name__ == "__main__":
    main()
