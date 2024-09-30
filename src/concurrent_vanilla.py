import os
import importlib.util
from multiprocessing import Pool, Manager
from generate.pddl import VanillaSampling
from utils import TraceSearchTimeOut

sample_max_time = 60

def write_to_file(output_data, file_path):
    try:
        with open(file_path, 'a', buffering=1) as file:  # Line buffered
            for output in output_data:
                file.write(output)
    except Exception as e:
        print(f"Error writing to file {file_path}: {e}")

def generate_trace(args):
    domain, domain_name, problem_name, domain_file_path, problem_file_path = args
    try:
        print(f"Generating traces for: Domain: {domain_name}, Problem: {problem_name}")

        vanilla = VanillaSampling(domain_file_path, problem_file_path, plan_len=50, num_traces=1, max_time=sample_max_time)
        traces = vanilla.generate_traces()

        output_data = []
        for i, trace in enumerate(traces):
            t = []
            for action in trace.actions:
                a = [action.name]
                for obj in action.obj_params:
                    a.append(obj.name)
                t.append(f"({' '.join(a)})")
            output = f"{domain['name']}&&{domain_name}&&{problem_name}&&{50}&&{','.join(t)}\n"
            output_data.append(output)

        return output_data

    except TimeoutError as te:
        print(f"Timeout generating traces: {te}")
    except TraceSearchTimeOut as tst:
        print(f"Timeout trace search: {tst}")
    except Exception as e:
        print(f"Error generating traces: {e}")
    return None

def main():
    # Get the list of directories in the 'data/classical' folder
    dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'classical'))
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'plain_traces'))

    directories = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    directories.sort()

    # Create a Manager to handle the lock
    with Manager() as manager:
        lock = manager.Lock()
    # Prepare the list of tasks
        tasks = []
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

                # Prepare tasks for parallel processing
                for domain in domains:
                    if (domain['name'] not in ['agricola']):
                        continue
                    for problem in domain['problems']:
                        domain_file, problem_file = problem
                        domain_name = domain_file.split('.')[0]
                        problem_name = problem_file.split('.')[0]
                        domain_file_path = os.path.join(dir, domain_file)
                        problem_file_path = os.path.join(dir, problem_file)

                        tasks.append((domain, domain_name, problem_name, domain_file_path, problem_file_path))

    # Use multiprocessing Pool to process the tasks
    
        with Pool(processes=10) as pool:
            results = pool.map(generate_trace, tasks)
                
        for output_data in results:
            if output_data:
                write_to_file(output_data, os.path.join(out_dir, "vanilla.txt"))


if __name__ == "__main__":
    main()
