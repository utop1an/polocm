

import os
import argparse
from planner import RandomPlanner
import logging
import multiprocessing

def generate_trace(solution_dir, task_dir, planner):
    try:
        plan = parse_plan_file(solution_dir, planner)
        random_traces = planner.generate_traces()
    except Exception as e:
        logging.error(e)
        return None, None
    return plan, random_traces

def parse_plan_file(file, planner):
    with open(file, 'r') as f:
        lines = f.readlines()
    plan = []
    for line in lines:
        if line.startswith(";"):
            break
        action = line.strip().strip("()").split(" ")
        action_name = action[0]
        op = planner.task.get_action(action_name)
        assert op is not None, f"Action {action_name} not found in domain"

        args = action[1:]
        arg_types = [p.type_name for p in op.parameters]
        arg_with_types = [arg + "?"+ t for arg, t in zip(args, arg_types)]
        new_action = f"({action_name} {' '.join(arg_with_types)})"
        plan.append(new_action)
    return plan


def process_domain(domain, solution_dir, task_dir, trace_length, num_traces, rdPlannerTimeout, seed, max_objects):
    logging.info(f"Generating traces for domain: {domain}")
    domain_filepath = os.path.join(task_dir, domain, "domain.pddl")

    traning_dir = os.path.join(solution_dir, domain, "training/easy")
    output_data = []

    for plan_file in os.listdir(traning_dir):
        if not plan_file.endswith(".plan"):
            continue
        task_file = plan_file.replace(".plan", ".pddl")
        plan_filepath = os.path.join(traning_dir, plan_file)
        task_filepath = os.path.join(task_dir, domain, "training/easy", task_file)

        if not os.path.exists(task_filepath):
            continue

        planner = RandomPlanner(domain_filepath, task_filepath, plan_len=trace_length, num_traces=num_traces, max_time=rdPlannerTimeout, seed=seed)
        planner.initialize_task()
        number_of_objects = len(planner.task.objects)
        if max_objects is not None and number_of_objects > max_objects:
            continue

        plan, random_walk = generate_trace(plan_filepath, task_filepath, planner)
        if plan is None or random_walk is None:
            continue
        plan_data = f"{domain}&&{'plan'}&&{task_file}&&{'easy'}&&{number_of_objects}&&{len(plan)}&&{','.join(plan)}\n"
        output_data.append(plan_data)
        for trace in random_walk:
            trace_data = f"{domain}&&{'rand'}&&{task_file}&&{'easy'}&&{number_of_objects}&&{len(trace)}&&{','.join(trace)}\n"
            output_data.append(trace_data)

    logging.info(f"{domain} done...")
    return output_data

def main(args):
    input_path = args.i
    output_path = args.o
    trace_length = args.l
    seed = args.s
    rdPlannerTimeout = args.t
    num_traces = args.n
    max_objects = args.m

    logging.basicConfig(level=logging.INFO)
    logging.info(f"Generating traces from raw plans in {input_path}")

    if not os.path.exists(input_path):
        logging.error(f"Input path {input_path} does not exist")
        return

    solution_dir = os.path.join(input_path, "solutions")
    task_dir = os.path.join(input_path, "tasks")

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    output_file = os.path.join(output_path, "plain_traces.txt")

    # Use multiprocessing Pool to parallelize domain processing
    with multiprocessing.Pool(processes=8) as pool:
        # Collect all domains first
        domains = [domain for domain in os.listdir(solution_dir) if os.path.isdir(os.path.join(solution_dir, domain))]
        
        # Map each domain to the process_domain function
        results = pool.starmap(
            process_domain,
            [(domain, solution_dir, task_dir, trace_length, num_traces, rdPlannerTimeout, seed, max_objects) for domain in domains],
            chunksize=1
        )

    logging.info(f"Writting traces to {output_file}")
    # Write the results to the output file
    with open(output_file, 'w', buffering=1) as file:  # Line buffered
        for result in results:
            for line in result:
                file.write(line)

    
    logging.info(f"Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse plain traces from raw plans and generate random traces")
    parser.add_argument("--i", type=str, default="./data/raw_traces", help="Directory containing raw plans and task pddls")
    parser.add_argument("--o", type=str, default="./data/plain_traces", help="Output file path")
    parser.add_argument("--l", type=int, default=50, help="Length of the random traces")
    parser.add_argument("--s", type=int, default=42, help="Seed for random planner")
    parser.add_argument("--t", type=int, default=30, help="Timeout for random planner generating traces per task in seconds")
    parser.add_argument("--n", type=int, default=1, help="Number of random traces per task")
    parser.add_argument("--m", type=int, default=None, help="Maximum number of objects in the tasks")
    args = parser.parse_args()
    main(args)
