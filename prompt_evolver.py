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
    def mutate_prompt(self, failed_prompt, problem, wrong_answer):

        short_feedback = str(wrong_answer)[:50]
        meta_prompt = f"""
        You are a prompt engineer, an AI model failed to solve a logic puzzle using a specific prompt:
        -Failed prompt: {failed_prompt}
        -The AI give a Wrong answer starting with: {short_feedback}
        
        Your task is to write a NEW, improved prompt to help AI to reason better.
        MANDATORY RULES:
        1)DON'T BE SPECIFIC: do not mention specific items, names, colors from puzzle (e.g no cars, no birds, no color, no name).
        2)PROCEDURAL: you have to write a command about how to think (e.g "build a mental map", "Chain of thought").
        3)OUTPUT FORMAT: you must instruct the AI to end its response with the exact chosen option
        4)BREVITY: Keep the new prompt under 25 words.
        5)TAGS: wrap the prompt in <prompt> and </prompt>.
        Correct Example:
        <prompt>Analyze the spatial constraints step-by-step and determine the final sequence of all elements. Write only the answer.</prompt>
        """
        
        raw_response = self.llm_client.prompt_model(meta_prompt, max_new_tokens = 50)
    
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


        