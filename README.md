# Information Propagation Simulation
This project simulates the spread of information (specifically, belief in certain topics) through a population using agent-based modeling. Agents interact with each other, update their beliefs based on conversations, and potentially "infect" others with their opinions.

## Project Structure
├── main.py                 # Main simulation runner
├── agent.py                # Citizen agent implementation
├── prompt.py               # LLM prompt templates and examples
├── traits.py               # Agent trait generation utilities
├── world.py                # World model and simulation logic
├── requirements.txt        # Python dependencies
└── README.md               # This documentation


## Key Components
1. **Agent Model (agent.py)**
   - Defines the Citizen agent class with:
     - Personal attributes (name, age, traits, qualification)
     - Health states (Susceptible, Infected, Recovered)
     - Opinion/belief tracking mechanisms
     - Interaction and belief update logic using LLM prompts

2. **World Simulation (world.py)**
   - Manages the simulation environment with:
     - Agent initialization and scheduling
     - Interaction dynamics (contact rate)
     - Disease propagation mechanics
     - Data collection and checkpointing

3. **LLM Integration (prompt.py)**
   - Contains prompt templates for:
     - Opinion updates with confirmation bias modeling
     - Memory summarization (short-term and long-term)
     - Multilingual support (English/Chinese)
     - Example belief statements for different topics

4. **Utilities (traits.py)**
   - Generates realistic agent characteristics:
     - Random names from global datasets
     - Big Five personality traits
     - Educational qualifications
     - OpenAI API helpers for LLM interactions

## How to Run
### Install dependencies:
```bash
pip install -r requirements.txt

Prepare a CSV file containing topics (with clean_title column)
Run the simulation:
python main.py   --name "SimulationRun"   --contact_rate 3   --no_init_healthy 28   --no_init_infect 2   --no_days 4   --no_of_runs 1
Configuration Options
| Parameter           | Description                     | Default   |
| ------------------- | ------------------------------- | --------- |
| `--name`            | Run identifier for output files | "ProFNSE" |
| `--contact_rate`    | Daily interactions per agent    | 3         |
| `--no_init_healthy` | Initial susceptible agents      | 28        |
| `--no_init_infect`  | Initial infected agents         | 2         |
| `--no_days`         | Simulation duration (days)      | 4         |
| `--no_of_runs`      | Number of simulation runs       | 1         |
| `--offset`          | Checkpoint load offset          | 0         |
| `--load_from_run`   | Checkpoint run identifier       | 0         |
Output
Generates propagation_feature.csv containing:
Topic being discussed
Propagation rate (belief spread speed)
Propagation depth (total infected)
Daily infection counts
Key Features
🧠 Cognitive modeling with confirmation bias
🌐 Multilingual prompt support
💾 Checkpointing for long simulations
📊 Detailed propagation metrics
🔄 Customizable interaction dynamics
🧪 Personality trait generation
Dependencies
Python 3.7+
Mesa (agent-based modeling)
Pandas (data processing)
names-dataset (name generation)
OpenAI API (LLM interactions)
Sample Topic

The default simulation uses housing policy claims like:
"In Liuzhou, you can apply for affordable rental housing with just 30,000 yuan"

Agents demonstrate either belief (infected) or skepticism (susceptible) about these statements.
Customization

To simulate different topics:
Modify topic_sentence_infected and topic_sentence_susceptible in prompt.py
Update the prompt templates for different domains
Adjust agent trait distributions in traits.py
