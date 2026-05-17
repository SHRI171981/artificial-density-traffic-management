import os
import pandas as pd
from scripts.yolov8_training import count_objects
from config import MODEL_PATH
import random

random.seed(10)


class TrafficController:
    def __init__(self, alpha=1, sim_percent = 0):
        self.ad_density = {}
        self.alpha = alpha
        self.sim_percent = sim_percent

    def _return_density_range(self, density):
        diff = int(density * self.sim_percent / 100)
        return random.randint(density - diff, density + diff)

    def evaluate_cycle(self, images_dir):
        actual_density = {}
        valid_extensions = ('.jpg', '.jpeg', '.png')
        image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(valid_extensions)]
        
        for img_file in image_files:
            lane = os.path.splitext(img_file)[0]
            img_path = os.path.join(images_dir, img_file)
            
            density = count_objects(MODEL_PATH, img_path) 
            actual_density[lane] = self._return_density_range(density)
            
            if lane not in self.ad_density:
                self.ad_density[lane] = 0

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
        }

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