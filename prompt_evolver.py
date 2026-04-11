import re

class PromptEvolver:
    def __init__(self,data_manager,llm_client):
        self.data_manager = data_manager
        self.llm_client = llm_client
        self.current_prompt = "Solve this logic problem"
    
    #take the answer and evaluate it
    def evaluate_answer(self, answer, correct_answer):
        if correct_answer.lower() in answer.lower():
            #Answer correct, give full point
            return 1
        return 0
    
    #If the answer is wrong, then we need to feed the prompt to a model and make it perform better
    def mutate_prompt(self, failed_prompt, problem, wrong_answer):
        meta_prompt = f"""
        Analyze this error:

        -Problem: {problem}
        -Failed prompt: {failed_prompt}
        -Wrong answer: {wrong_answer}
        
        You ONLY need to generate the new prompt without adding anything
        RULE:
        1)Do not generate anything but the new prompt
        2)Do not write code
        3)Write opening with <New prompt> and ending with </New prompt>
        """
        
        raw_response = self.llm_client.prompt_model(meta_prompt)
    
        # Estrazione robusta del contenuto tra i tag
        match = re.search(r'<prompt>(.*?)</prompt>', raw_response, re.DOTALL)
        if match:
            return match.group(1).strip()
    
        # Fallback: se il modello fallisce i tag, restituisci comunque la risposta pulita
        return raw_response.strip()
    
    def run_evolution(self, steps = 3):
        print(f"Prompt optimization cycle")

        for i in range(steps):
            #Get an example from the dataset
            sample = self.data_manager.get_random_sample()
            full_input = f"{self.current_prompt} \n\n {sample['question']}"

            #feed the example to the model
            answer = self.llm_client.prompt_model(full_input)

            #evaluate answer 
            score = self.evaluate_answer(answer, sample['correct_answer'])
            
            print(f"step {i+1} | score: {score}")

            #if score is 0, then mutate the next prompt
            if score < 1:
                print("There was an error. New prompt generation")
                self.current_prompt = self.mutate_prompt(self.current_prompt,sample['question'],answer)
                print(f"New prompt: {self.current_prompt}")
            else:
                print("Correct answer")

        return self.current_prompt


        