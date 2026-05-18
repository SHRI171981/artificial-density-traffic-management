import os
import json
import pandas as pd
from ultralytics import YOLO
from scripts.yolov8_training import count_objects
from config import MODEL_PATH
import random

class TrafficController:
    def __init__(self, alpha=1, sim_percent = 0):
        self.ad_density = {
            "A": 0,
            "B": 0,
            "C": 0,
            "D": 0,
        }
        self.alpha = alpha
        self.sim_percent = sim_percent
        self.model = YOLO(MODEL_PATH)
        with open("saved_image_path.json") as f:
            self.saved_images = json.loads(f.read())

    def _return_density_range(self, density):
        diff = int(density * self.sim_percent / 100)
        return random.randint(density - diff, density + diff)

    def _return_random_image(self):
        return {
            "A": random.choice(self.saved_images['A']),
            "B": random.choice(self.saved_images['B']),
            "C": random.choice(self.saved_images['C']),
            "D": random.choice(self.saved_images['D'])
        }
    
    def _return_actual_density(self):
        image_files = self._return_random_image()
        return {
            "A": count_objects(model=self.model, image_path=image_files["A"]),
            "B": count_objects(model=self.model, image_path=image_files["B"]),
            "C": count_objects(model=self.model, image_path=image_files["C"]),
            "D": count_objects(model=self.model, image_path=image_files["D"]),
        }, image_files

    def evaluate_cycle(self):
        actual_density, image_files = self._return_actual_density()

        # Capture state prior to mutation to ensure logging math aligns with execution logic
        applied_ad_weight = self.ad_density.copy()

        total_density = {}
        for lane in actual_density:
            total_density[lane] = actual_density[lane] + applied_ad_weight[lane]

        cur_lane = max(total_density, key=total_density.get)
        cleared_density = actual_density[cur_lane]

        for lane in self.ad_density:
            if lane == cur_lane:
                self.ad_density[lane] = 0
            else:
                self.ad_density[lane] += cleared_density * self.alpha

        return {
            'green_lane': cur_lane,
            'actual_counts': actual_density,
            'applied_ad_weight': applied_ad_weight, 
            'effective_totals': total_density
        }, image_files

if __name__ == "__main__":
    TARGET_DIR = "./test_images"
    
    # Initialize controller once to persist state across continuous polling intervals
    controller = TrafficController(sim_percent=10)
    
    # Execute the evaluation cycle 25 times sequentially
    for cycle in range(1, 26):
        print(f"--- Cycle {cycle} ---")
        decision_matrix = controller.evaluate_cycle(TARGET_DIR)
        
        df_results = pd.DataFrame({
            'Physical_Count': decision_matrix['actual_counts'],
            'Artificial_Weight': decision_matrix['applied_ad_weight'],
            'Effective_Total': decision_matrix['effective_totals']
        })
        
        print(df_results)
        print(f"ROUTING DECISION: Green Light assigned to Lane {decision_matrix['green_lane']}\n")
        print("\n")