from vis_nav_game import Player, Action, Phase
import pygame
import cv2
import numpy as np
import os
import pickle
from sklearn.cluster import KMeans
from sklearn.neighbors import BallTree
from tqdm import tqdm
from natsort import natsorted
import logging
from collections import deque

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


class AutonomousNavigator(Player):
    def __init__(self):
        self.save_dir = "data/final_data/images_subsample"
        if not os.path.exists(self.save_dir):
            raise FileNotFoundError(f"Directory {self.save_dir} does not exist")


        self.sift = cv2.SIFT_create()

        self.sift_descriptors = None
        self.codebook = None
        self.database = None
        self.tree = None
        self.exploration_images = []
        self.stuck_cooldown = 0          # steps until we allow another stuck trigger
        self.best_dist_to_goal = None    # best (smallest) distance so far
        self.no_progress_steps = 0       # steps since last improvement

        self.fpv = None
        self.goal_id = None
        self.current_id = None
        self.num_images = 0

        self.position_history = deque(maxlen=20)
        self.action_history = deque(maxlen=5)
        self.consecutive_forward = 0
        self.recovery_queue = deque()


        self.show_visualization = True
        self.visualization_window = None
        self.target_comparison_window = None

        self.lazy_indexing = True
        self.vlad_cache_dir = os.path.join(os.getcwd(), "vlad_cache")
        os.makedirs(self.vlad_cache_dir, exist_ok=True)

        super(AutonomousNavigator, self).__init__()

        if os.path.exists("sift_descriptors.npy"):
            self.sift_descriptors = np.load("sift_descriptors.npy")
        if os.path.exists("codebook.pkl"):
            with open("codebook.pkl", "rb") as f:
                self.codebook = pickle.load(f)

        logging.info("initialized")

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

    def compute_sift_features(self):
        files = natsorted([x for x in os.listdir(self.save_dir) if x.endswith('.jpg')])
        self.num_images = len(files)
        sift_descriptors = []

        logging.info(f"Computing SIFT features for {self.num_images} images...")
        for img_file in tqdm(files, desc="Extracting SIFT"):
            img = cv2.imread(os.path.join(self.save_dir, img_file))
            self.exploration_images.append(img)
            _, des = self.sift.detectAndCompute(img, None)
            if des is not None:
                sift_descriptors.extend(des)

        return np.asarray(sift_descriptors)

    def get_vlad(self, img):
        _, des = self.sift.detectAndCompute(img, None)

        if des is None or len(des) == 0:
            return np.zeros(self.codebook.n_clusters * 128)
        pred_labels = self.codebook.predict(des)
        centroids = self.codebook.cluster_centers_
        k = self.codebook.n_clusters

        vlad_feature = np.zeros([k, des.shape[1]])

        for i in range(k):
            if np.sum(pred_labels == i) > 0:
                vlad_feature[i] = np.sum(des[pred_labels == i, :] - centroids[i], axis=0)

        vlad_feature = vlad_feature.flatten()
        vlad_feature = np.sign(vlad_feature) * np.sqrt(np.abs(vlad_feature))
        norm = np.linalg.norm(vlad_feature)
        if norm > 0:
            vlad_feature = vlad_feature / norm

        return vlad_feature

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

    def compute_turn_direction(self, current_img, target_img):
        kp1, des1 = self.sift.detectAndCompute(current_img, None)
        kp2, des2 = self.sift.detectAndCompute(target_img, None)

        if des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10:
            return 0


        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        matches = bf.knnMatch(des1, des2, k=2)

        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

        if len(good_matches) < 10:
            return 0

        x_displacements = []
        for match in good_matches:
            pt1_x = kp1[match.queryIdx].pt[0]
            pt2_x = kp2[match.trainIdx].pt[0]
            displacement = pt2_x - pt1_x
            x_displacements.append(displacement)

        median_displacement = np.median(x_displacements)


        if median_displacement > 25:
            return 1
        elif median_displacement < -25:
            return -1
        else:
            return 0

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

    def pre_navigation(self):
        super(AutonomousNavigator, self).pre_navigation()

        files = natsorted([x for x in os.listdir(self.save_dir) if x.endswith('.jpg')])
        self.num_images = len(files)
        logging.info("Loading exploration images...")
        self.exploration_images = []
        for img_file in tqdm(files, desc="Loading images"):
            img = cv2.imread(os.path.join(self.save_dir, img_file))
            self.exploration_images.append(img)

        if self.codebook is None:
            if self.sift_descriptors is None:
                logging.info("Computing SIFT features for codebook...")
                self.sift_descriptors = self.compute_sift_features()
                np.save("sift_descriptors.npy", self.sift_descriptors)
            logging.info("Building codebook (K-means clustering)...")
            self.codebook = KMeans(
                n_clusters=128,
                init='k-means++',
                n_init=5,
                verbose=0
            ).fit(self.sift_descriptors)
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
        indices, distances = self.get_neighbor(self.fpv, k=1)
        self.current_id = indices[0]
        confidence = distances[0]
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


if __name__ == "__main__":
    import vis_nav_game
    print("*" * 30)
    print("dvl2013 Auto Target Planner")
    print("*" * 30)
    player = AutonomousNavigator()
    vis_nav_game.play(the_player=player)
