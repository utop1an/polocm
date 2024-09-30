import os
import importlib.util
from multiprocessing import Pool, Manager
from planner.random_planner import RandomPlanner
from utils import TraceSearchTimeOut

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
        plan_len = 50
        rd_planner = RandomPlanner(domain_file_path, problem_file_path, plan_len=plan_len, num_traces=1, max_time=45)
        traces = rd_planner.generate_traces()
        
        output_data = []
        for i, trace in enumerate(traces):
            t = [op.name for op in trace]
            output = f"{domain['name']}&&{domain_name}&&{problem_name}&&{plan_len}&&{','.join(t)}\n"
            output_data.append(output)

        return output_data

    except TimeoutError as te:
        print(f"Timeout generating traces: {te}")
        output = f"{domain['name']}&&{domain_name}&&{problem_name}&&{plan_len}&&TimeoutError\n"
        return [output]
    except TraceSearchTimeOut as tst:
        print(f"Timeout trace search: {tst}")
        output = f"{domain['name']}&&{domain_name}&&{problem_name}&&{plan_len}&&TraceSearchTimeOut\n"
        return [output]
    except Exception as e:
        print(f"Error generating traces: {e}")
        output = f"{domain['name']}&&{domain_name}&&{problem_name}&&{plan_len}&&Error\n"
        return [output]

def main():
    dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'classical'))
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'plain_traces'))

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    directories = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    directories.sort()

    # Create a Manager to handle the lock
    with Manager() as manager:
        lock = manager.Lock()

        # Prepare tasks for parallel processing
        tasks = []
        for directory in directories:
            api_file_path = os.path.join(dir, directory, 'api.py')

            if os.path.exists(api_file_path):
                spec = importlib.util.spec_from_file_location('api', api_file_path)
                api = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(api)

                domains = api.domains
                for domain in domains:
                    if domain['name'] not in ['agricola']:
                        continue
                    for problem in domain['problems']:
                        domain_file, problem_file = problem
                        domain_name = domain_file.split('.')[0]
                        problem_name = problem_file.split('.')[0]
                        domain_file_path = os.path.join(dir, domain_file)
                        problem_file_path = os.path.join(dir, problem_file)

                        tasks.append((domain, domain_name, problem_name, domain_file_path, problem_file_path))

        # Use Pool to process tasks in parallel
        with Pool(processes=10) as pool:
            results = pool.map(generate_trace, tasks)

        # Write results to files
        for output_data in results:
            if output_data:
                write_to_file(output_data, os.path.join(out_dir, "test.txt"))

if __name__ == "__main__":
    main()

