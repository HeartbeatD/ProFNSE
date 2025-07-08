
from world import World
import argparse
import pandas as pd
from matplotlib import pyplot as plt
import os
import requests

url = ""
topics_file = r""
topics_df = pd.read_csv(topics_file)
topics = topics_df['clean_title'].tolist()  # Convert the title column to a list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="ProFNSE", help="Name of the run to save outputs.")
    parser.add_argument("--contact_rate", default=3, type=int, help="Contact Rate")
    parser.add_argument("--no_init_healthy", default=28, type=int,
                        help="Number of initial healthy people in the world.")
    parser.add_argument("--no_init_infect", default=2, type=int,
                        help="Number of initial infected people in the world.")
    parser.add_argument("--no_days", default=4, type=int,
                        help="Total number of days the world should run.")
    parser.add_argument("--no_of_runs", default=1, type=int, help="Total number of times you want to run this code.")
    parser.add_argument("--offset", default=0, type=int,
                        help="offset is equal to number of days if you need to load a checkpoint")
    parser.add_argument("--load_from_run", default=0, type=int,
                        help="equal to run # - 1 if you need to load a checkpoint (e.g. if you want to load run 2 checkpoint 8, then offset = 8, load_from_run = 1)")
    args = parser.parse_args()
    # Ensure the output directory exists

    # List to store all results
    all_results = []
    for i, topic in enumerate(topics):
        print(f"Processing topic {i + 1}/{len(topics)}: {topic}")

        # Create a World instance. Other parameters of args are omitted here and need to be added according to the actual situation.
        model = World(args, initial_healthy=args.no_init_healthy, initial_infected=args.no_init_infect,
                      contact_rate=args.contact_rate, topic=topic)

        # Running the model
        model.run_model(checkpoint_path='', offset=0)

        # Obtain the number of infections per day, and calculate the infection rate and depth of infection
        daily_infected_list = model.daily_infected_list
        # Propagation_rate = (daily_infected_list[-1])-2
        Propagation_rate = ((daily_infected_list[0] - 2) + (
                daily_infected_list[1] - daily_infected_list[0]) + (
                                    daily_infected_list[2] - daily_infected_list[1]) + (
                                    daily_infected_list[3] - daily_infected_list[2])) / 4
        Propagation_depth = daily_infected_list[-1]
        # Add the results to the list
        all_results.append({
            'Topic': topic,
            'Propagation Rate': Propagation_rate,
            'Propagation Depth': Propagation_depth,
            'daily_infected_list': daily_infected_list
        })
        # Convert all_results to a DataFrame
        results_df = pd.DataFrame(all_results)

        # Split daily_infected_list into multiple columns
        daily_infected_df = results_df['daily_infected_list'].apply(pd.Series)

        # Rename the split columns
        daily_infected_df = daily_infected_df.rename(columns=lambda x: f'daily_infected_{x + 1}')

        # Merge the split columns back into the original DataFrame
        results_df = pd.concat([results_df.drop('daily_infected_list', axis=1), daily_infected_df], axis=1)

        # Save the DataFrame to a CSV file
        output_dir = r""
        filename = "propagation_feature.csv"
        output_path = os.path.join(output_dir, filename)


        os.makedirs(output_dir, exist_ok=True)


        results_df.to_csv(output_path, index=False)

