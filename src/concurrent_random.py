import os
import importlib.util
import json
import signal
from concurrent.futures import ProcessPoolExecutor, as_completed
from planner.random_planner import RandomPlanner
from utils import TraceSearchTimeOut

def handler(signum, frame):
    raise TimeoutError("Execution time exceeded")

def generate_trace(domain, domain_name, problem_name, domain_file_path, problem_file_path):
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(40)
    try:
       

        print(f"Generating traces for: Domain: {domain_name}, Problem: {problem_name}")

        rd_planner = RandomPlanner(domain_file_path, problem_file_path, plan_len=50, num_traces=1)
        traces = rd_planner.generate_traces()
        plan_len = 50
        output_data = []
        for i, trace in enumerate(traces):
            t = [op.name for op in trace]
            output = f"{domain['name']}&&{domain_name}&&{problem_name}&&{plan_len}&&{','.join(t)}\n"
            output_data.append(output)

        return output_data

    except TimeoutError as te:
        print(f"Timeout generating traces: {te}")
    except TraceSearchTimeOut as tst:
        print(f"Timeout trace search: {tst}")
    except Exception as e:
        print(f"Error generating traces: {e}")
    finally:
        signal.alarm(0)
    return None

def main():
    # Get the list of directories in the 'data/classical' folder
    dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'classical'))
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'json_traces'))

    directories = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    directories.sort()

    with open(os.path.join(out_dir, "random.txt"), 'a') as file:
        # Prepare tasks for parallel processing
        tasks = []
        with ProcessPoolExecutor(max_workers=os.cpu_count()-2) as executor:
            # For each directory
            for directory in directories:
                # Construct the path to the api.py file
                api_file_path = os.path.join(dir, directory, 'api.py')

                # Check if the api.py file exists
                if os.path.exists(api_file_path):
                    # Load the api.py module
                    spec = importlib.util.spec_from_file_location('api', api_file_path)
                    api = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(api)

                    # Access the 'domains' variable
                    domains = api.domains

                    # Schedule tasks for parallel processing
                    for domain in domains:
                        for problem in domain['problems']:
                            domain_file, problem_file = problem
                            domain_name = domain_file.split('.')[0]
                            problem_name = problem_file.split('.')[0]
                            domain_file_path = os.path.join(dir, domain_file)
                            problem_file_path = os.path.join(dir, problem_file)

                            tasks.append(executor.submit(generate_trace, domain, domain_name, problem_name, domain_file_path, problem_file_path))

            # Process the results as they are completed
            for future in as_completed(tasks):
                output_data = future.result()
                if output_data:
                    for output in output_data:
                        file.write(output)

if __name__ == "__main__":
    main()
