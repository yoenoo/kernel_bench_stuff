"""
Bunch of helpful functions for evaluation
"""
import requests

def fetch_kernel_from_database(run_name: str, problem_id: int, sample_id: int, server_url: str):
    """
    Intenral to us with our django database
    Return a dict with kernel hash, kernel code, problem_id
    """
    response = requests.get(
        f"{server_url}/get_kernel_by_run_problem_sample/{run_name}/{problem_id}/{sample_id}",
        json={"run_name": run_name, "problem_id": problem_id, "sample_id": sample_id},
    )
    assert response.status_code == 200
    response_json = response.json()
    assert str(response_json["problem_id"]) == str(problem_id)
    return response_json["kernel"]

# if __name__ == "__main__":
#     fetch_kernel_from_database("kernelbench_prompt_v2_level_2", 1, 1, "http://localhost:9091")




