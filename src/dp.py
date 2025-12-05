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
    def __init__(self):
        self.save_dir = "data/final_data/images_subsample"
        if not os.path.exists(self.save_dir):
            raise FileNotFoundError(f"Directory {self.save_dir} does not exist")
        if os.path.exists("codebook.pkl"):
            with open("codebook.pkl", "rb") as f:
                self.codebook = pickle.load(f)
        self.DATA_ROOT = "./data/final_data/"
        #QUERY_DIR = os.path.join(DATA_ROOT, "query")
        self.DB_DIR = os.path.join(self.DATA_ROOT, "images_subsample")
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone = None
        self.yolo_descriptors = None
        self.codebook = None
        self.database = None
        self.exploration_images = []
        self.stuck_cooldown = 0          # steps until we allow another stuck trigger
        self.best_dist_to_goal = None    # best (smallest) distance so far
        self.no_progress_steps = 0       # steps since last improvement
        self.db_index= None
        self.fpv = None
        self.goal_id = None
        self.current_id = None
        self.num_images = 0
        self.lazy_indexing = True
        self.position_history = deque(maxlen=20)
        self.action_history = deque(maxlen=5)
        self.consecutive_forward = 0
        self.recovery_queue = deque()


        self.show_visualization = True
        self.visualization_window = None
        self.target_comparison_window = None
        super(Autonnomous_resnet_Navigator, self).__init__()

        if os.path.exists("yoloo_descriptors.npy"):
            self.yolo_descriptors = np.load("yolo_descriptors.npy")
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

    @torch.no_grad()
    def compute_descriptor(self, backbone, img_):
        #img = Image.open(img_path).convert("RGB")
        #img = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
        x = preprocess(img_).unsqueeze(0).to(self.DEVICE)  
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
        if len(self.position_history) < 15:
            return False

        if self.stuck_cooldown > 0:
            return False

        recent = list(self.position_history)[-15:]
        unique = len(set(recent))

        from collections import Counter
        counts = Counter(recent)
        most_common_id, most_common_count = counts.most_common(1)[0]
        freq = most_common_count / len(recent)

        stuck_pos = (unique <= 4 and freq > 0.7)

        if self.best_dist_to_goal is None or dist_to_goal < self.best_dist_to_goal - 2:
            self.best_dist_to_goal = dist_to_goal
            self.no_progress_steps = 0
        else:
            self.no_progress_steps += 1

        stuck_progress = self.no_progress_steps > 40

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
        #    indices, distances = self.get_neighbor(target, k=1)
        #    goal_candidates.append((indices[0], distances[0]))
        #    matched_images.append(self.exploration_images[indices[0]])
        #    logging.info(f"Target view {i}: ID {indices[0]} (dist: {distances[0]:.4f})")
            matches = self.retrieve_matches(self.backbone, self.db_index, targets, top_k=1)
            for i, (score, distance, fname) in enumerate(matches, start=1):
                m_img = Image.open(os.path.join(self.DB_DIR, fname)).convert("RGB")
                matched_images.append(m_img)
            goal_candidates.append((matches[0], matches[1]))
        #for rank, (score, fname) in enumerate(matches, start=1):
        #    print(f"#{rank}: {fname} (similarity = {score:.4f})")
        #good_matches = []
        #for i, (score, distance, fname) in enumerate(matches, start=1):
            #m_img = Image.open(os.path.join(self.DB_DIR, fname)).convert("RGB")
            
        #    good_matches.append(m_img)
        goal_id = matches[0]
        logging.info(f"Primary goal: Image {goal_id}")
        if self.show_visualization:
            self.show_target_comparison(targets, matched_images, goal_candidates)

        return goal_id
    def retrieve_matches(self, backbone, db_index, img, top_k=10):
        #for i, fname in enumerate(img_path):
        #    path = os.path.join(self.DB_DIR, fname)
        q_desc = self.compute_descriptor(backbone, img)
     
        sims = []
        for fname, d_desc in db_index:
            s = float(np.dot(q_desc, d_desc)) 
            dist =  np.linalg.norm(q_desc, d_desc)
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

    def compute_turn_direction(self, current_img):
        #kp1, des1 = self.sift.detectAndCompute(current_img, None)
        #kp2, des2 = self.sift.detectAndCompute(target_img, None)

        #if des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10:
        #    return 0


        #bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        #matches = bf.knnMatch(des1, des2, k=2)

        matches = self.retrieve_matches(self.backbone, self.db_index, current_img, top_k=20)

        #for rank, (score, fname) in enumerate(matches, start=1):
        #    print(f"#{rank}: {fname} (similarity = {score:.4f})")
        #good_matches = []
        #for i, (score, distance, fname) in enumerate(matches, start=1):
            #m_img = Image.open(os.path.join(self.DB_DIR, fname)).convert("RGB")
            
        #    good_matches.append(m_img)

        #if len(good_matches) < 10:
        #    return 0

        #for match_pair in matches:
        #    if len(match_pair) == 2:
        #        m, n = match_pair
        #        if m.distance < 0.75 * n.distance:
        #            good_matches.append(m)

        #if len(good_matches) < 10:
        #    return 0

        #x_displacements = []
        #for match in good_matches:
        #    torch.dist(,p=2)
        #for match in good_matches:
        #    pt1_x = kp1[match.queryIdx].pt[0]
        #    pt2_x = kp2[match.trainIdx].pt[0]
        #    displacement = pt2_x - pt1_x
        #    x_displacements.append(displacement)

        median_displacement = np.median(matches.dist)

        if median_displacement > 25:
            return 1
        elif median_displacement < -25:
            return -1
        else:
            return 0
        
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

    def select_action(self):
        if self.fpv is None:
            return Action.IDLE
        #indices, distances = self.get_neighbor(self.fpv, k=1)

        matched_images = []
        #for i, target in enumerate(targets):
        #    indices, distances = self.get_neighbor(target, k=1)
        #    goal_candidates.append((indices[0], distances[0]))
        #    matched_images.append(self.exploration_images[indices[0]])
        #    logging.info(f"Target view {i}: ID {indices[0]} (dist: {distances[0]:.4f})")
        matches = self.retrieve_matches(self.backbone, self.db_index, self.fpv, top_k=1)
        #for i, (score, distance, fname) in enumerate(matches, start=1):
        #    m_img = Image.open(os.path.join(self.DB_DIR, fname)).convert("RGB")
        #    matched_images.append(m_img)
        #goal_candidates.append((matches[0], matches[1]))
        #for rank, (score, fname) in enumerate(matches, start=1):
        #    print(f"#{rank}: {fname} (similarity = {score:.4f})")
        #good_matches = []
        #for i, (score, distance, fname) in enumerate(matches, start=1):
            #m_img = Image.open(os.path.join(self.DB_DIR, fname)).convert("RGB")
            
        #    good_matches.append(m_img)
        #goal_id = matches[0]

        self.current_id = matches[0]
        confidence = matches[1]
        self.position_history.append(self.current_id)

        dist_to_goal = abs(self.goal_id - self.current_id)
        if len(self.position_history) % 25 == 0:
            dist_to_goal = abs(self.goal_id - self.current_id)
            logging.info(
                f"Position: {self.current_id}, Goal: {self.goal_id}, "
                f"Distance: {dist_to_goal}, Confidence: {confidence:.4f}"
            )
        if abs(self.current_id - self.goal_id) <= 2:
            logging.info("Goal reached! Checking in...")
            return Action.CHECKIN
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
            return Action.FORWARD
        if self.recovery_queue:
            action = self.recovery_queue.popleft()
            self.action_history.append(action)
            if self.show_visualization:
                self.show_navigation_visualization(self.fpv, target_img, action,
                                                   self.current_id, target_id)
            return action

        if self.is_stuck(dist_to_goal):
            logging.warning("Stuck detected! Initiating back-off and centering...")
            self.consecutive_forward = 0
            self.stuck_cooldown = 5
            self.position_history.clear()
            self.recovery_queue.extend([Action.BACKWARD, Action.BACKWARD, Action.BACKWARD])
            self.recovery_queue.append(Action.FORWARD)
            self.recovery_queue.extend([Action.RIGHT, Action.FORWARD, Action.LEFT, Action.FORWARD])
            action = self.recovery_queue.popleft()
            self.action_history.append(action)
            if self.show_visualization:
                self.show_navigation_visualization(self.fpv, target_img, action,
                                                   self.current_id, target_id)
            return action
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
        if self.consecutive_forward == 0 and len(self.action_history) >= 3:
            recent_actions = list(self.action_history)[-3:]
            if all(a in [Action.LEFT, Action.RIGHT] for a in recent_actions):
                logging.info("Too many turns, forcing forward")
                action = Action.FORWARD
                self.consecutive_forward = 1
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
    
    def pre_navigation(self):
        super(Autonnomous_resnet_Navigator, self).pre_navigation()

        files = natsorted([x for x in os.listdir(self.save_dir) if x.endswith('.jpg')])
        self.num_images = len(files)
        logging.info("Loading exploration images...")
        self.exploration_images = []
        for img_file in tqdm(files, desc="Loading images"):
            img = cv2.imread(os.path.join(self.save_dir, img_file))
            self.exploration_images.append(img)

        if self.codebook is None:

            if self.yolo_descriptors is None:
                logging.info("Computing Yolo features for codebook...")
                self.yolo_descriptors = self.compute_descriptor()
                np.save("yolo_descriptors.npy", self.yolo_descriptors)
            logging.info("Building codebook (K-means clustering)...")
            self.codebook = KMeans(
                n_clusters=128,
                init='k-means++',
                n_init=5,
                verbose=0
            ).fit(self.yolo_descriptors)
            
            
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

    vis_nav_game.play(the_player=model)

if __name__ == "__main__":
    main()
