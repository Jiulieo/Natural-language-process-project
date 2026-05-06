import re
import string

class PromptEvolver:
    def __init__(self,data_manager,llm_client):
        self.data_manager = data_manager
        self.llm_client = llm_client
        self.current_prompt = "Solve this logic problem carefully following the instruction"
    
    #take the answer and evaluate it
    def evaluate_answer(self, answer, correct_answer):
        def clean_text(text):
            text = text.lower()
            text = text.translate(str.maketrans('','',string.punctuation))
            text = text.replace("the ", "").replace("a ", "").replace("an ","")
            return "".join(text.split())

        clean_correct = clean_text(correct_answer)
        clean_model = clean_text(answer)

        if clean_correct in clean_model:
            #Answer correct, give full point
            return 1.0
        return 0.0
    
    #If the answer is wrong, then we need to feed the prompt to a model and make it perform better
    def mutate_prompt(self, failed_prompt, problem, wrong_answer):

        short_feedback = str(wrong_answer)[:50]
        meta_prompt = f"""You are a Prompt Engineer. Your task is to write a short, universal instruction for a logic puzzle solver.

        FAILED PROMPT: "{failed_prompt}"

        Write a new, improved instruction. It must be ONE sentence.
        The 1.8B model gets confused if it reasons too much. Instruct it to be CONCISE and direct.

        BAD EXAMPLES (Causes hallucinations):
        - <prompt>Break down the problem into manageable steps and reason systematically.</prompt>
        - <prompt>Analyze the sequence of letters and numbers step-by-step.</prompt>

        GOOD EXAMPLES (Forces direct, concise answers):
        - <prompt>Read the constraints carefully and output ONLY the final correct statement.</prompt>
        - <prompt>Determine the correct order and write the final answer in a single sentence.</prompt>
        - <prompt>Solve the logic puzzle and directly state the final choice without extra reasoning.</prompt>

        NEW PROMPT:
        <prompt>"""

        #to be more rigid in the generation we lower the temperature
        raw_response = self.llm_client.prompt_model(meta_prompt, max_new_tokens = 100, temperature = 0.1)
    
        # use regex to extract only prompt if done correctly
        match = re.search(r'<prompt>(.*?)</prompt>', raw_response, re.DOTALL)
        if match:
            return match.group(1).strip()
    
        return raw_response.strip()
    
    def run_evolution(self, steps = 5):
        print(f"Prompt optimization cycle")

        best_prompt = self.current_prompt
        best_score = -1
        best_answer_length = float('inf')

        history_lengths = []
        history_scores = []

        for i in range(steps):
            #Get an example from the dataset
            sample = self.data_manager.get_random_sample()
            full_input = f"{self.current_prompt} \n\n {sample['question']}"

            #feed the example to the model
            answer = self.llm_client.prompt_model(full_input, max_new_tokens = 512)
            current_answer_len = len(answer)

            #Save data to plot
            history_lengths.append(current_answer_len)

            #evaluate answer 
            score = self.evaluate_answer(answer, sample['correct_answer'])
            
            print(f"step {i+1} | score: {score}")
            print(f"Target corretto: {sample['correct_answer']}")
            print(f"Risposta generata dal modello:\n{answer}")

            #To update we use the fact that it pass the singular test
            if score > best_score:
                best_score = score
                best_prompt = self.current_prompt
                best_answer_length = current_answer_len
            elif score == best_score and score == 1.0 and current_answer_len < best_answer_length:
                # It tied for the best score (a perfect 1.0), AND it was more efficient
                best_prompt = self.current_prompt
                best_answer_length = current_answer_len

            
            #if score is 0, then mutate the next prompt
            if score < 1:
                print("There was an error. New prompt generation")
                self.current_prompt = self.mutate_prompt(self.current_prompt,sample['question'],answer)
                print(f"New prompt: {self.current_prompt}")
            else:
                print("Correct answer")

        #return like that to store in a dictionary
        return {
            "best_prompt": best_prompt,
            "lengths_over_time": history_lengths,
            "scores_over_time": history_scores
        }


        