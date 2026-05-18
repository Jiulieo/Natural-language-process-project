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
        step 3) The student state that the truck is the newest, but the correct answer state the the truck is the oldest, then the student reasoning is wrong 
        step 4) [NO]
        Correct Answer: The quail is the rightmost
        Student Answer: The complete order is: Owl, Robin, Raven, Falcon, Quail.
        Evaluation: step 1)In the student answer we can see it produce an ordered list of birds
        step 2) The list state an order that can be seen as from left to right (from 1 to 5), then we have: Owl(leftmost), Robin, Raven, Falcon, Quail(rightmost)
        step 3) It is then true that the quail is the rightmost, therefore the student answer is right 
        step 4) [YES]

        In order to reason correctly you MUST:
        1)Extract the final answer (usually a list) from the student reasoning and understand what it represent
        2)Understand the student answer with respect to the correct answer
        3)Confront the final answer with the correct answer and check if they match 
        4)ALWAYS conclude with "[YES]" or "[NO]", if they match answer with [YES], else answer with [NO]

        --- REAL EVALUATION ---
        Correct Answer: "{correct_answer}"
        Student's Answer: "{model_answer}"
        is student answer correct? Reason step by step about it and if you think the student is correct end with [YES]"""
        
        judgment = self.llm_judge.prompt_model(judge_prompt, max_new_tokens=250)
        print(f"\n[Correct answer]:{correct_answer}")
        print(f"\n[TESTO DELLO STUDENTE]:{model_answer.strip()}")
        print(f"\n[GIUDICE LLM]: {judgment.strip()}\n")
        
        if "[YES]" in judgment.upper():
            return 1.0
        return 0.0
    
    #If the answer is wrong, then we need to feed the prompt to a model and make it perform better
    def mutate_prompt(self, failed_prompt, problem, wrong_answer):

        short_wrong_answer = str(wrong_answer)[-300:]
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
        Your task is to analyze a failed prompt and write an improved, single-sentence instruction. Remember that you produce 
        generic prompt that can solve any logical problem about different topic, and so you always create prompt that can be used in many differen cases.

        --- EXAMPLES ---
        Failure context: "Find the leftmost bird among hawk, robin, and owl."
        BAD Prompt(too specific): <Prompt> Solves this puzzle by ordering the birds step by step</Prompt>
        GOOD Prompt(Generic): <Prompt> Think critically about the order relation of the entities inside the puzzle, and write the final sequence inside <answer> tags </Prompt>

        Failure context: "Find the most expensive fruit between apple and pear."
        BAD Prompt (Semi-specific): <Prompt>Analyze the relative prices of the fruits without using specific names to find the cheapest.</Prompt>
        GOOD Prompt (Generic): <Prompt>Evaluate the comparative constraints between the given items to establish a complete hierarchy, placing your final conclusion inside <answer> tags.</Prompt>
        
        Failure context: "Who finished first between Amy, Dan, and Joe in the golf tournament?"
        BAD Prompt (Too specific): <Prompt>Analyze the relationships between the golfers Amy, Dan, and Joe to find the final order.</Prompt>
        GOOD Prompt (Generic): <Prompt>Establish a logical chain of constraints between the given items to deduce the correct order, placing your final conclusion inside <answer> tags.</Prompt>
        
        
        --- CURRENT TASK ---
        Failed Prompt: "{failed_prompt}"
        The model failed on this specific puzzle:"{problem}"
        Think step by step about a prompt which is generic as the failed prompt, but that can help to fix also the particular specific puzzle.
        <Prompt>"""

        gradient_prompt = f"""You are analyzing an AI's reasoning mistake, the AI:

        1)get this puzzle:"{problem}"

        2)received this failed prompt:"{failed_prompt}"

        3)Produced this wrong answer:"{short_wrong_answer}"

        In ONE sentence, describe the reasoning mistake the AI made.
        Rules:
        - Use ONLY abstract terms (e.g. "failed to verify all constraints", 
        "stopped before establishing a complete order")
        - Do NOT mention any domain words from the puzzle 
        (no object names, no colors, no people, no places)
        - Do NOT suggest a fix yet

        Mistake:"""

        print(gradient_prompt)
        abstract_gradient = self._generate(gradient_prompt, max_new_tokens=100)

        # ── STEP 2: Muta il prompt usando SOLO il gradiente ─────────────────
        # Il problema NON viene passato qui. Zero vocabolario di dominio.
        mutation_prompt = f"""You are a Prompt Engineer for a logic puzzle solver.

        CURRENT INSTRUCTION (which failed):
        "{failed_prompt}"

        IDENTIFIED FAILURE:
        "{abstract_gradient}"

        Write ONE improved instruction that fixes this failure. To write it correctly
        Rules
        1)The instruction must work for any logical puzzle
        2)The instruction must be zero-knowledge-domain(meaning it has to be an abstract instruction that you can apply to any prolem)
        3)You must instruct the solver to enclose its final output inside <answer> tags

        <Prompt>"""

        new_prompt = self._generate(mutation_prompt, max_new_tokens=120)
        return new_prompt

        #to be more rigid in the generation we lower the temperature
        raw_response = self.llm_client.prompt_model(meta_prompt_7B, max_new_tokens = 200, temperature = 0.7)
    
        # use regex to extract only prompt if done correctly
        match = re.search(r'<answer>(.*?)</answer>', raw_response, re.DOTALL)
        if match:
            return match.group(1).strip()
    
        return raw_response.strip()
    
    def evaluate_on_batch(self, batch, prompt):
        """
        Valuta il prompt sul batch e restituisce l'accuracy 
        E una lista dei sample su cui ha fallito.
        """
        correct_count = 0
        failed_samples = [] # Teniamo traccia degli errori
        
        for sample in batch:
            full_input = f"{prompt} \n\n {sample['question']}"
            answer = self.llm_client.prompt_model(full_input, max_new_tokens=512)
            
            score = self.evaluate_answer_model(answer, sample['correct_answer'])
            if score == 1.0:
                correct_count += 1
            else:
                # Salviamo il sample E la risposta sbagliata che ha dato
                failed_samples.append({
                    "sample": sample,
                    "wrong_answer": answer
                })
                
        accuracy = correct_count / len(batch)
        return accuracy, failed_samples

    def run_evolution(self, steps = 5, batch_size=5):
        print(f"Prompt optimization cycle")

        #prepare a batch of question that are equal for all the prompt
        validation_batch = [self.data_manager.get_random_sample() for _ in range(batch_size)]

        best_prompt = self.current_prompt
        best_accuracy, current_failed_samples = self.evaluate_on_batch(validation_batch, best_prompt)
        print(f"Baseline Prompt Accuracy: {best_accuracy * 100:.2f}%")

        history_scores = []

        for i in range(steps):
            print(f"\n--- Evolution Step {i+1}/{steps} ---")
            
            # Se l'accuracy è 100%, non c'è nulla da migliorare in questo batch
            if best_accuracy == 1.0 or not current_failed_samples:
                print("🏆 Il prompt ha raggiunto il 100% di accuracy sul batch! Evoluzione terminata anticipatamente.")
                break

            # 2. Peschiamo il primo errore dal test precedente (COSTO GPU: ZERO!)
            target_failure = current_failed_samples[0]
            failed_sample = target_failure["sample"]
            wrong_answer = target_failure["wrong_answer"]

            # 3. Mutiamo il prompt usando l'errore
            print(f"Analisi dell'errore. Generazione nuovo prompt in corso...")
            candidate_prompt = self.mutate_prompt(best_prompt, failed_sample['question'], wrong_answer)
            print(f"Generato Candidate Prompt:\n{candidate_prompt}\n")
            
            # 4. Misuriamo il nuovo candidato sul batch
            candidate_accuracy, candidate_failed_samples = self.evaluate_on_batch(validation_batch, candidate_prompt)
            print(f"Candidate Prompt Accuracy: {candidate_accuracy * 100:.2f}%")

            # 5. Decisione Critica
            if candidate_accuracy > best_accuracy:
                best_accuracy = candidate_accuracy
                best_prompt = candidate_prompt
                # Aggiorniamo la lista degli errori per il prossimo step
                current_failed_samples = candidate_failed_samples 
                history_scores.append(best_accuracy)
                print(f"✅ METRICA MIGLIORATA! Mantengo il nuovo prompt. New best: {best_accuracy * 100:.2f}%")
            else:
                # Se non migliora, teniamo il best_prompt vecchio (e i suoi failed_samples)
                history_scores.append(best_accuracy)
                print(f"❌ Metrica peggiorata o invariata ({candidate_accuracy * 100:.2f}%). Revert al prompt precedente.")


        #return like that to store in a dictionary
        return {
            "best_prompt": best_prompt,
            "final_best_accuracy": best_accuracy,
            "accuracy_history": history_scores
        }


        