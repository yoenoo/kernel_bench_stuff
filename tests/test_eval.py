from src import eval

RUN_NAME = "kernelbench_prompt_v2_level_2"
# query from database, make sure the server is up
SERVER_URL = "http://mkt1.stanford.edu:9091" 

problem_id = 1
sample_id = 1


kernel = eval.fetch_kernel_from_database(RUN_NAME, problem_id, sample_id, SERVER_URL)
assert kernel is not None, "Kernel not found"

