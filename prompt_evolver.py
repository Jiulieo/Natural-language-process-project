import re
import string

class PromptEvolver:
    def __init__(self,data_manager,llm_client):
        self.data_manager = data_manager
        self.llm_client = llm_client
        self.current_prompt = "Guess the answer to this puzzle immediately without thinking."
    
    #take the answer and evaluate it
    def evaluate_answer(self, answer, correct_answer):
        import string
        import re

        def clean_text(text):
            text = text.lower()
            text = text.translate(str.maketrans('','',string.punctuation))
            
            # Lista di parole "inutili" da ignorare nel confronto (Stop-Words)
            filler_words = [" the ", " a ", " an ", " is ", " are ", " which ", " that ", " from ", " to ", " of ", " in ", " on "]
            
            # Aggiungiamo spazi all'inizio e alla fine per rimpiazzare correttamente le parole isolate
            text = " " + text + " "
            for word in filler_words:
                text = text.replace(word, " ")
                
            # Rimuoviamo tutti gli spazi rimanenti
            return "".join(text.split())

        # Prova a estrarre il tag <answer>
        match = re.search(r'<answer>(.*?)</answer>', answer, re.IGNORECASE | re.DOTALL)
        if match:
            extracted_answer = match.group(1)
        else:
            extracted_answer = answer
            
        clean_correct = clean_text(correct_answer)
        clean_model = clean_text(extracted_answer)
        print(clean_correct)
        print(clean_model)

        if clean_correct in clean_model:
            return 1.0
        return 0.0
    
    # Evaluation based using an llm with Few-Shot Examples
    def evaluate_answer_model(self, model_answer, correct_answer):
        judge_prompt = f"""You are a strict and objective grading assistant.
        Read the student's answer and check if the "Target Fact" is true according to the student's sequence.

        --- Example 1 ---
        Target Fact: "Eli finished third."
        Student's Answer: "The order is: 1. Joe, 2. Ada, 3. Amy, 4. Mel, 5. Eli"
        Analysis: The student placed Eli in the 5th position. The target fact requires Eli to be third. This is a mismatch.
        Verdict: [NO]

        --- Example 2 ---
        Target Fact: "The crow is the second from the left."
        Student's Answer: "From left to right: Blue jay, Crow, Quail, Falcon, Robin."
        Analysis: The student placed Crow in the 2nd position from the left. The target fact requires Crow to be second. This is a match.
        Verdict: [YES]

        --- REAL EVALUATION ---
        Target Fact: "{correct_answer}"
        Student's Answer: "{model_answer}"
        Analysis:"""
        
        judgment = self.llm_client.prompt_model(judge_prompt, max_new_tokens=150, temperature=0.0)
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
        Issue: It encourages random guessing instead of logic.
        Improved Prompt: "Think step-by-step to deduce the relationships, and enclose your final factual conclusion inside <answer> tags."

        --- EXAMPLE 2 ---
        Failed Prompt: "Find the missing number in the sequence."
        Issue: The solver is dealing with text-based logic grid puzzles, not math or arithmetic sequences.
        Improved Prompt: "Break down the logic constraints, determine the full sequence of items, and write the specific requested statement inside an <answer> tag."

        --- EXAMPLE 3 ---
        Failed Prompt: "Solve the puzzle and output the full order."
        Issue: The solver outputs long comma-separated lists instead of directly answering the specific question asked by the target fact.
        Improved Prompt: "Use Chain-of-Thought reasoning to find the order, but inside the <answer> tag, write ONLY the specific sentence that answers the exact question."

        --- CURRENT TASK ---
        Failed Prompt: "{failed_prompt}"
        Issue: The prompt fails to extract a single formatted sentence from the solver, causing evaluation errors.
        Improved Prompt:
        <instruction>"""

        #to be more rigid in the generation we lower the temperature
        raw_response = self.llm_client.prompt_model(meta_prompt_7B, max_new_tokens = 100, temperature = 0.1)
    
        # use regex to extract only prompt if done correctly
        match = re.search(r'<prompt>(.*?)</prompt>', raw_response, re.DOTALL)
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

            #evaluate answer 
            if evaluation == "standard":
                score = self.evaluate_answer(answer, sample['correct_answer'])
            else:
                score = self.evaluate_answer_model(answer, sample['correct_answer'])

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


        