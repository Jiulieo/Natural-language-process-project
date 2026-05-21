import json
import os
import random

class DatasetManager():
    #initialize the manager to point the folder with the example i want to read
    def __init__(self,task_folder_path,train_ratio=0.8, seed=42):
        self.file_path = os.path.join(task_folder_path, "task.json")
        
        # Fixed seed, to ensure reproducibility
        random.seed(seed)
        
        all_data = self.load_data()
        
        # --- Dataset split ---
        random.shuffle(all_data)
        split_index = int(len(all_data) * train_ratio)
        
        self.train_set = all_data[:split_index]
        self.test_set = all_data[split_index:]
    
    def load_data(self):
        #read json
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            return raw_data.get("examples",[])
        except FileNotFoundError:
            print(f"File non trovato: in {self.file_path}")
            return []
        
    def get_random_train_sample(self):
        # Draw a problem from train set
        if not self.train_set: return None
        sample = random.choice(self.train_set)
        return self._format_sample(sample)
    
    def get_random_test_sample(self):
        # Draw random problem from test set
        if not self.test_set: return None
        sample = random.choice(self.test_set)
        return self._format_sample(sample)

    def _format_sample(self, sample):
        question = sample.get("input", "")
        target = sample.get("target_scores", {})

        correct = ""
        options = []

        for option, p in target.items():
            options.append(option)
            if p == 1:
                correct = option

        return {
            "question": question,
            "options": options,
            "correct_answer": correct
        }
