The project should develop a prompt optimizer to solve logical problem which is composed of three modules:
-A data manager which load data and then draw random sample (random problem) from the dataset loaded (BIG bench was used and their work can be found 
on github to)
-A llm_client which is the module that get a prompt and make the model generate an answer
-The prompt_evolver, which evaluate the answer and, if wrong, try to develop a better prompt.
