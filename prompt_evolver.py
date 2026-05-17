import re
import string

class PromptEvolver:
    def __init__(self,data_manager,llm_client,llm_judge):
        self.data_manager = data_manager
        self.llm_client = llm_client
        self.llm_judge = llm_judge
        self.current_prompt = "Guess the answer to this puzzle immediately without thinking."

    
    # Evaluation based using an llm with Few-Shot Examples
    def evaluate_answer_model(self, model_answer, correct_answer):
        judge_prompt = f"""You are a strict evaluator who have to decide if a student is reasoning correctly. The student will not always answer clearly,
        you have to first think very well about the correct answer you receive and understand if the student answered correctly to that answer.
        At the end of your evaluation you have to write [YES] if student was correct.

        ---Bad Examples---
        Correct Answer: kiwis are the most expensive
        Student Answer: The order of fruit prices, from cheapest to most expensive, is: 1. Cantaloupes 2. Apples 3. Loquats 4. Watermelons 5. Kiwis
        Evaluation: [NO]
        Correct Answer: Eli finished second to last
        Student Answer: Final Answer: Eve: 1st Eli: 2nd Ana: 3rd Rob: 4th Mya: 5th
        Evaluation: [YES]

        ---Good examples---
        Correct Answer: The truck is the oldest
        Student Answer: The order from oldest to newest is: Motorcycle, Hatchback, Station Wagon, Convertible, Truck
        Evaluation: step 1) The student answer is an ordered list of vehicles
        step 2) The list state an order from oldest to newest so we can say: Motorcycle(oldest), hatchback, station wagon, Convertible, Truck(Newest)
        step 3) The student state that the truck is the newest, but the correct answer state the the truck is the oldest, then the student reasoning is wrong [NO]
        Correct Answer: The quail is the rightmost
        Student Answer: The complete order is: Owl, Robin, Raven, Falcon, Quail.
        Evaluation: 1)In the student answer we can see it produce an ordered list of birds
        2) The list state an order that can be seen as from left to right (from 1 to 5), then we have: Owl(leftmost), Robin, Raven, Falcon, Quail(rightmost)
        3) It is then true that the quail is the rightmost, therefore the student answer is right [YES]

        In order to reason correctly you MUST:
        1)Extract the final answer (usually a list) from the student reasoning and understand what it represent
        2)Understand the student answer with respect to the correct answer
        3)Confront the final answer with the correct answer and check if they match, if they match answer with [YES], else answer with [NO]

        --- REAL EVALUATION ---
        Correct Answer: "{correct_answer}"
        Student's Answer: "{model_answer}"
        is student answer correct? Reason step by step about it and if you think the student is correct end with [YES]"""
        
        judgment = self.llm_judge.prompt_model(judge_prompt, max_new_tokens=250)
        print(f"\n[GIUDICE LLM]: {judgment.strip()}\n")
        
        if "[YES]" in judgment.upper():
            return 1.0
        return 0.0
    
    #If the answer is wrong, then we need to feed the prompt to a model and make it perform better
    def mutate_prompt(self, failed_prompt, problem, wrong_answer):

        short_feedback = str(wrong_answer)[:50]
        meta_prompt_1_8B = f"""You are a Prompt Engineer. Your task is to write a short, universal instruction for a logic puzzle solver.

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

        #Because the 7B parameter can be a little more verbose i use another meta prompt here that let him reason more
        meta_prompt_7B = f"""You are an expert Prompt Engineer for a logic puzzle solver.
        Your task is to analyze a failed prompt and write an improved, single-sentence instruction.

        --- EXAMPLE 1 ---
        Failed Prompt: "Guess the answer immediately."
        Issue: It encourages random guessing instead of step-by-step logic.
        Improved Prompt: "Think step-by-step to logically deduce the order of all items, and then enclose your final factual conclusion strictly inside <answer> tags."

        --- EXAMPLE 2 ---
        Failed Prompt: "Solve the puzzle and output the full order."
        Issue: The solver outputs long lists and skips reasoning. It needs to use Chain-of-Thought to find the sequence, but only output the final fact.
        Improved Prompt: "Use Chain-of-Thought reasoning to find the complete sequence, but inside the <answer> tag, write ONLY the specific sentence that answers the exact target question."

        --- EXAMPLE 3 ---
        Failed Prompt: "Write a single clear sentence in the answer tag."
        Issue: The prompt formats the answer correctly, but forgets to tell the model to reason first. Without explicitly saying "think step-by-step", the solver will guess.
        Improved Prompt: "Employ step-by-step logical deduction to solve the constraints, and enclose ONLY the exact requested statement inside an <answer> tag."

        --- CURRENT TASK ---
        Failed Prompt: "{failed_prompt}"
        Issue: The prompt fails because it does not explicitly instruct the model to think step-by-step or use Chain-of-Thought before writing the formatted answer.
        Improved Prompt:
        <Prompt>"""

        #to be more rigid in the generation we lower the temperature
        raw_response = self.llm_client.prompt_model(meta_prompt_7B, max_new_tokens = 100, temperature = 0.1)
    
        # use regex to extract only prompt if done correctly
        match = re.search(r'<answer>(.*?)</answer>', raw_response, re.DOTALL)
        if match:
            return match.group(1).strip()
    
        return raw_response.strip()
    
    def run_evolution(self, steps = 5,evaluation = "standard"):
        print(f"Prompt optimization cycle")

        best_prompt = self.current_prompt
        best_score = -1
        best_answer_length = -1

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

            score = self.evaluate_answer_model(answer, sample['correct_answer'])
            history_scores.append(score)

            print(f"step {i+1} | score: {score}")
            print(f"Target corretto: {sample['correct_answer']}")
            print(f"Risposta generata dal modello:\n{answer}")

            # To update we use the fact that it pass the singular test
            if score > best_score:
                best_score = score
                best_prompt = self.current_prompt
                best_answer_length = current_answer_len
            elif score == best_score and score == 1.0:
                # Se usiamo la valutazione LLM (modelli grandi), premiamo la verbosità (CoT)
                if evaluation == "model" and current_answer_len > best_answer_length:
                    best_prompt = self.current_prompt
                    best_answer_length = current_answer_len
                # Se usiamo la valutazione standard (modelli piccoli), premiamo la sintesi
                elif evaluation == "standard" and current_answer_len < best_answer_length:
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


        