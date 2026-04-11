import re

class PromptEvolver:
    def __init__(self,data_manager,llm_client):
        self.data_manager = data_manager
        self.llm_client = llm_client
        self.current_prompt = "Solve this logic problem carefully following the instruction"
    
    #take the answer and evaluate it
    def evaluate_answer(self, answer, correct_answer):
        if correct_answer.lower() in answer.lower():
            #Answer correct, give full point
            return 1
        return 0
    
    #If the answer is wrong, then we need to feed the prompt to a model and make it perform better
    def mutate_prompt(self, failed_prompt, wrong_answer):
        meta_prompt = f"""
        You are a prompt engineer, an AI model failed to solve a logic puzzle using a specific prompt:
        -Failed prompt: {failed_prompt}
        -Wrong answer: {wrong_answer}
        
        Your task is to write a NEW, improved prompt to help AI to reason better.
        MANDATORY RULES:
        1)BE GENERIC: do not mention specific items from puzzle (e.g no cars, no birds, no color)
        2)METHODOLOGICAL: order the AI to use a specific technique (e.g "build a mental map", "Chain of thought", "List the items first")
        3)FORMAT: you must wrap your new prompt between <prompt> and </prompt> tags.
        Example:
        <prompt> Analyze the problem step by step to find the correct order, then, write the correct answer as it appear in the option. </prompt>
        """
        
        raw_response = self.llm_client.prompt_model(meta_prompt)
    
        # use regex to extract only prompt if done correctly
        match = re.search(r'<prompt>(.*?)</prompt>', raw_response, re.DOTALL)
        if match:
            return match.group(1).strip()
    
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


        