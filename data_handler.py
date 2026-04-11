import json
import os
import random

class DatasetManager():
    #initialize the manager to point the folder with the example i want to read
    def __init__(self,task_folder_path):
        self.file_path = os.path.join(task_folder_path, "task.json")
        self.data = self.load_data()
    
    def load_data(self):
        #legge il json
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            return raw_data.get("examples",[])
        except FileNotFoundError:
            print(f"File non trovato: in {self.file_path}")
            return []
        
    def get_random_sample(self):
        #Pesca un problema a caso e restituisce input e target
        if not self.data:
            return None
        #qui prende dalla lista (caricata tramite json.load) un elemento a caso, che corrisponde ad una coppia input/target
        sample = random.choice(self.data)

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
