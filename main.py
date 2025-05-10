import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F # Import F for functional API
import numpy as np
import random
from collections import deque, namedtuple
import math
import os
import sys # Added for SUMO_HOME check
import traci # Corrected import for TraCI

# --- Configuration & Hyperparameters (from paper & user clarifications) ---

# Environment/General
STATE_SIZE = 8 # Assuming 8 lanes as per self.road_ids tuple
ACTION_SIZE = 24 # User clarification: 4 ways * 6 duration choices (5 to 30 in steps of 5)

NUM_EPISODES = 100 # Paper: 100 episodes
STEPS_PER_EPISODE = 500 # Paper: 500 steps
# Note: Total simulation steps will be NUM_EPISODES * STEPS_PER_EPISODE

# SSEN Specific
SIGMA = 0.2 # Optimal sigma from paper for SSE_DQN
COMPRESSION_N = 2 # Denoted as N in paper
FC_UNITS = 256 # Fully connected layer units
GRU_UNITS = FC_UNITS // COMPRESSION_N # 128
COMPRESSED_VALUE_UNITS = FC_UNITS // COMPRESSION_N # 128

# DQN Agent
MAX_MEMORY_SIZE = 10000
BATCH_SIZE = 64
GAMMA = 0.95 # Discount factor
LEARNING_RATE = 0.001
TARGET_NETWORK_UPDATE_FREQUENCY = 100 # steps
EPSILON_MAX = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY_RATE = 0.02 # The '0.02' value from paper, interpreted as linear decay factor per episode

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- SSEN Network (Self-Sequential Estimator Network) ---
class SSEN(nn.Module):
    def __init__(self, state_size, action_size, fc_units=FC_UNITS,
                 compressed_units=COMPRESSED_VALUE_UNITS, gru_units=GRU_UNITS, sigma_val=SIGMA):
        super(SSEN, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.sigma = sigma_val
        self.gru_units = gru_units # Store for h_0 initialization

        # Shared initial layer
        # Paper: "softsign for the first hidden layer"
        self.shared_fc1 = nn.Linear(state_size, fc_units)

        # V(s) stream
        # Paper: "In compressed value... values fall within the interval of -1 to 1." (due to softsign)
        self.compress_fc = nn.Linear(fc_units, compressed_units) # Outputs the "compressed value"

        # Paper: "ReLU (rectified linear unit) for the last hidden."
        # Assuming a hidden layer before V output, consistent with A(s,a) stream.
        self.v_hidden_fc = nn.Linear(compressed_units, compressed_units)
        self.v_output_fc = nn.Linear(compressed_units, 1) # Scalar V(s)

        # A(s,a) stream
        # Paper: "sequential approach utilizing GRU"
        self.gru = nn.GRU(fc_units, gru_units, batch_first=True) # batch_first=True expects (batch, seq, feature)

        # Decoder part for A(s,a) from "New equation"
        # Paper: "Linear + Activation function" after "New equation". Assuming ReLU.
        self.a_hidden_fc = nn.Linear(gru_units, gru_units)
        self.a_output_fc = nn.Linear(gru_units, action_size)

        self._initialize_weights()

    def _initialize_weights(self):
        # Paper: "Xavier/Glorot initialization [7] for ReLU function is used"
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2: # For Linear and GRU weights
                    nn.init.xavier_uniform_(param)
                # else: # For biases (or 1D weights if any) - biases are handled next
            elif 'bias' in name:
                nn.init.zeros_(param) # Biases are often initialized to zero

    def forward(self, state, h_0=None):
        # state: (batch_size, state_size)
        # h_0: initial hidden state for GRU (num_layers * num_directions, batch, hidden_size) -> (1, batch_size, gru_units)

        batch_size = state.size(0)

        # Shared path
        # Corrected softsign usage
        x_shared = F.softsign(self.shared_fc1(state)) # (batch_size, fc_units)

        # V(s) stream
        # Paper: "In compressed value... values fall within the interval of -1 to 1." (due to softsign)
        # Corrected softsign usage
        compressed_value = F.softsign(self.compress_fc(x_shared)) # (batch_size, compressed_units)

        v_hidden = F.relu(self.v_hidden_fc(compressed_value))
        V_s = self.v_output_fc(v_hidden) # (batch_size, 1)

        # A(s,a) stream
        x_gru_input = x_shared.unsqueeze(1) # (batch_size, 1, fc_units) - seq_len = 1

        if h_0 is None: # Create a default zero hidden state if not provided
            h_0 = torch.zeros(1, batch_size, self.gru_units).to(state.device)
        # Ensure h_0 batch size matches current input state's batch size if h_0 was provided
        elif h_0.size(1) != batch_size:
             h_0 = torch.zeros(1, batch_size, self.gru_units).to(state.device)


        gru_output, next_hidden_state = self.gru(x_gru_input, h_0)
        gru_output = gru_output.squeeze(1) # (batch_size, gru_units)

        # New equation = sequence_method_output / (sigma + log(2 + compressed_value))
        # compressed_value range is (-1, 1). So (2 + compressed_value) is (1, 3). Log is safe.
        log_comp_val = torch.log(2.0 + compressed_value) # (batch_size, compressed_units)

        # Denominator: sigma + log_comp_val. For sigma=0.2, log_comp_val is (0, ~1.09). Denom is (0.2, ~1.29).
        denominator = self.sigma + log_comp_val

        # Element-wise division. Ensure dimensions match if compressed_units != gru_units
        # Corrected the shape comparison here:
        if gru_output.shape != denominator.shape:
            # This case should not happen if COMPRESSED_VALUE_UNITS == GRU_UNITS
            # If they were different, broadcasting or tiling might be needed, or a design review.
            # For now, assuming they are equal as per interpretation.
            # The "New equation" diagram shows "Sequence method" and "New Equation" having same width as "Compress".
            pass

        new_eq_output = gru_output / denominator # (batch_size, gru_units)

        a_hidden = F.relu(self.a_hidden_fc(new_eq_output))
        A_sa = self.a_output_fc(a_hidden) # (batch_size, action_size)

        # Combine V(s) and A(s,a) using Dueling DQN formula
        Q_s_a = V_s + (A_sa - A_sa.mean(dim=1, keepdim=True))

        return Q_s_a, next_hidden_state


# --- Replay Buffer ---
Experience = namedtuple("Experience",
                        field_names=["state", "action", "reward", "next_state", "done"])

class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, device):
        self.action_size = action_size # Not strictly needed here but good for consistency
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.device = device

    def add(self, state, action, reward, next_state, done):
        e = Experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        # Convert numpy arrays to torch tensors
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        # 'done' flag needs to be float for multiplication in the loss calculation
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)


        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)


# --- SSE_DQNAgent ---
class SSE_DQNAgent:
    def __init__(self, state_size, action_size, device):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device

        self.policy_net = SSEN(state_size, action_size).to(device)
        self.target_net = SSEN(state_size, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = ReplayBuffer(action_size, MAX_MEMORY_SIZE, BATCH_SIZE, device)
        self.t_step = 0 # Counter for updating target network

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every TARGET_NETWORK_UPDATE_FREQUENCY time steps
        self.t_step = (self.t_step + 1) % TARGET_NETWORK_UPDATE_FREQUENCY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, epsilon=0., h_0=None):
        """Returns actions for given state as per current policy."""
        # Ensure state is a torch tensor and has batch dimension
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device) # (1, state_size)

        # Set network to evaluation mode for inference
        self.policy_net.eval()
        with torch.no_grad():
            # Get Q-values and next hidden state from the policy network
            q_values, next_h_0 = self.policy_net(state_t, h_0)
        # Set network back to training mode
        self.policy_net.train()

        # Epsilon-greedy action selection
        if random.random() > epsilon:
            # Select the action with the highest Q-value
            action = np.argmax(q_values.cpu().data.numpy())
        else:
            # Select a random action
            action = random.choice(np.arange(self.action_size))
        return action, next_h_0

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples."""
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        # Use default h_0 (zeros) for batch processing in the target network
        Q_targets_next_qvals, _ = self.target_net(next_states, h_0=None)
        Q_targets_next = Q_targets_next_qvals.detach().max(1)[0].unsqueeze(1) # Max Q-value for each next state

        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones)) # 1-dones handles terminal states

        # Get expected Q values from policy model
        # Use default h_0 (zeros) for batch processing in the policy network
        Q_expected_qvals, _ = self.policy_net(states, h_0=None)
        # Gather Q-values for the actions that were actually taken
        Q_expected = Q_expected_qvals.gather(1, actions)

        # Compute loss (Mean Squared Error)
        loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.optimizer.zero_grad() # Clear previous gradients
        loss.backward() # Compute gradients
        # Optional: Gradient clipping to prevent exploding gradients
        # torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1)
        self.optimizer.step() # Update weights

        # Update target network
        self._update_target_network()

    def _update_target_network(self):
        """Soft update model parameters: target_net = policy_net"""
        # In this implementation, we do a hard copy every TARGET_NETWORK_UPDATE_FREQUENCY steps
        self.target_net.load_state_dict(self.policy_net.state_dict())


# --- SUMO Environment ---
class SumoEnv:
    """
    Wrapper for SUMO environment using TraCI.
    Assumes specific lanes and a single traffic light 'J20'.
    """
    def __init__(self, sumo_cfg: str, max_steps: int = 1000):
        # Check if SUMO_HOME environment variable is set
        if 'SUMO_HOME' in os.environ:
            tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
            # Add the tools directory to the system path
            sys.path.append(tools)
        else:
            sys.exit("Please declare the environment variable 'SUMO_HOME'")

        # Corrected import for traci
        # import traci.cmd as traci_cmd # Removed
        # global traci # Removed
        # traci = traci_cmd # Removed


        self.sumo_cmd = [
            os.environ.get('SUMO_BINARY', 'sumo'), # Use sumo-gui if you want to visualize
            '-c', sumo_cfg,
            '--no-step-log', 'true',
            '--start', 'true',
            '--waiting-time-memory', str(max_steps) # Store waiting times for reward calculation
        ]
        self.max_steps = max_steps
        # Mapping action index to traffic light phase and duration
        # Action space 0-23 (4 ways * 6 durations)
        # Way 0: Actions 0-5 (Phase 0, durations 5, 10, 15, 20, 25, 30)
        # Way 1: Actions 6-11 (Phase 3, durations 5, 10, 15, 20, 25, 30)
        # Way 2: Actions 12-17 (Phase 6, durations 5, 10, 15, 20, 25, 30)
        # Way 3: Actions 18-23 (Phase 9, durations 5, 10, 15, 20, 25, 30)
        self.WAY_PHASES = {
            0: 0, # Phase index for Way 0 (e.g., N-S straight/left)
            1: 3, # Phase index for Way 1 (e.g., E-W straight/left)
            2: 6, # Phase index for Way 2 (e.g., N-S right)
            3: 9  # Phase index for Way 3 (e.g., E-W right)
        }
        self.YELLOW_DURATION = 3 # Standard yellow phase duration
        self.RED_DURATION = 2 # Standard red phase duration between phases

        # Using lane IDs instead of induction loop IDs
        self.lane_ids = ('E10_0', 'E10_1', '-E11_0', '-E11_1', '-E12_0', '-E12_1', 'E9_0', 'E9_1')
        # Ensure the number of lane IDs matches the STATE_SIZE
        if len(self.lane_ids) != STATE_SIZE:
            print(f"Warning: Number of lane IDs ({len(self.lane_ids)}) does not match STATE_SIZE ({STATE_SIZE}).")
            print("Please ensure your lane_ids tuple is correct for the state representation.")

        self.step_count = 0
        self.traffic_light_id = 'J20' # Assuming traffic light ID is 'J20'

        # Penalty configuration
        self.CONGESTION_THRESHOLD = 5 # Number of vehicles in a lane to trigger penalty
        self.PENALTY_PER_VEHICLE = -5.0 # Penalty value per vehicle over the threshold (Increased penalty)

    def reset(self):
        """Resets the SUMO simulation."""
        if traci.isLoaded():
            traci.close()
        # Start SUMO with the defined command
        traci.start(self.sumo_cmd)
        self.step_count = 0

        # Perform an initial simulation step to get the first state
        traci.simulationStep()
        self.step_count += 1

        return self._get_state()

    def step(self, action: int):
        """
        Applies the chosen action (traffic light phase and duration) and steps the simulation.
        Action is an integer from 0 to ACTION_SIZE - 1 (0-23).
        """
        # Decode action: way (0-3) and duration index (0-5)
        way = action // 6
        duration_index = action % 6 # 0 to 5
        green_dur = (duration_index + 1) * 5 # Duration in steps: 5, 10, 15, 20, 25, 30

        # Get the phase indices for the selected way
        green_phase = self.WAY_PHASES[way]
        # Assuming standard SUMO traffic light logic: Green -> Yellow -> Red
        # Need to find the corresponding yellow and red phases.
        # This often depends on the .net.xml definition.
        # A common pattern is Green (i), Yellow (i+1), Red (i+2) or similar.
        # Let's assume a simple pattern based on the provided SumoEnv structure:
        # Way 0 (Phase 0: Green) -> Phase 1 (Yellow) -> Phase 2 (Red)
        # Way 1 (Phase 3: Green) -> Phase 4 (Yellow) -> Phase 5 (Red)
        # ... and so on.
        yellow_phase = green_phase + 1
        red_phase = green_phase + 2 # This red phase typically clears the intersection before the next green

        # 1) Set green phase and run simulation for green_dur steps
        traci.trafficlight.setPhase(self.traffic_light_id, green_phase)
        for _ in range(green_dur):
            traci.simulationStep()
            self.step_count += 1

        # 2) Switch to yellow phase and run for YELLOW_DURATION
        traci.trafficlight.setPhase(self.traffic_light_id, yellow_phase)
        for _ in range(self.YELLOW_DURATION):
            traci.simulationStep()
            self.step_count += 1

        # 3) Switch to red phase and run for RED_DURATION (clearing phase)
        traci.trafficlight.setPhase(self.traffic_light_id, red_phase)
        for _ in range(self.RED_DURATION):
            traci.simulationStep()
            self.step_count += 1

        # Get state and compute reward after the phase cycle
        next_state = self._get_state()
        reward = self._compute_reward()

        # Check if episode is done
        done = self.step_count >= self.max_steps

        # Check if simulation crashed (optional but recommended)
        if not traci.isLoaded():
             print("SUMO simulation terminated unexpectedly.")
             done = True # Mark as done if simulation crashed

        return next_state, reward, done, {}

    def _get_state(self):
        """Gets the current state from lane data."""
        # State is the number of vehicles in each lane
        counts = [traci.lane.getLastStepVehicleNumber(lane_id) for lane_id in self.lane_ids]
        # Convert to float32 numpy array
        return np.array(counts, dtype=np.float32)

    def _compute_reward(self):
        """Computes the reward based on total waiting time and congestion penalty."""
        reward = 0.0
        # --- Waiting Time Penalty (Negative Reward) ---
        # Sum instantaneous waiting time for all vehicles on all relevant lanes
        for lid in self.lane_ids:
            reward -= traci.lane.getWaitingTime(lid)

        # --- Congestion Penalty ---
        # Add penalty if any lane has more than CONGESTION_THRESHOLD vehicles
        for lid in self.lane_ids: # Iterate through the specified lane IDs
            vehicle_count = traci.lane.getLastStepVehicleNumber(lid)
            if vehicle_count > self.CONGESTION_THRESHOLD:
                # Penalize for each vehicle over the threshold in this lane
                penalty_amount = (vehicle_count - self.CONGESTION_THRESHOLD) * self.PENALTY_PER_VEHICLE
                reward += penalty_amount # Add negative penalty to the reward

        return reward

    def close(self):
        """Closes the TraCI connection."""
        if traci.isLoaded():
            traci.close()


# --- Training Loop ---
def train_agent(sumo_cfg_path):
    # Initialize the SUMO environment
    # Total steps for the environment should cover all episodes and steps within them
    env = SumoEnv(sumo_cfg=sumo_cfg_path, max_steps=NUM_EPISODES * STEPS_PER_EPISODE)

    # Initialize the SSE-DQN agent with the correct state and action sizes
    agent = SSE_DQNAgent(state_size=STATE_SIZE, action_size=ACTION_SIZE, device=DEVICE)

    # List to store total reward per episode (optional, for monitoring)
    episode_rewards = []

    print("Starting SSE-DQN Training (SUMO Environment)...")

    try:
        for i_episode in range(1, NUM_EPISODES + 1):
            # Reset the environment for a new episode
            state = env.reset()
            # GRU hidden state for the current episode, batch_size=1 for acting
            current_episode_gru_h_state = torch.zeros(1, 1, GRU_UNITS).to(DEVICE)

            current_episode_total_reward = 0

            # Epsilon calculation based on paper's formula interpretation:
            # Linear decay from EPSILON_MAX to EPSILON_MIN over approx. 50 episodes
            epsilon = max(EPSILON_MIN, EPSILON_MAX - EPSILON_DECAY_RATE * (i_episode - 1))

            for t_step_in_episode in range(STEPS_PER_EPISODE):
                # Select action using the agent's policy
                action, next_episode_gru_h_state = agent.act(state, epsilon, current_episode_gru_h_state)

                # Execute the action in the SUMO environment
                next_state, reward, done, _ = env.step(action)

                # Store the experience and potentially trigger learning
                # 'done' from env.step indicates if the simulation reached max_steps
                # We also consider the end of the fixed episode length as 'done' for learning purposes
                done_for_learning = done or (t_step_in_episode == STEPS_PER_EPISODE - 1)

                agent.step(state, action, reward, next_state, done_for_learning)

                # Update state and hidden state for the next step
                state = next_state
                current_episode_gru_h_state = next_episode_gru_h_state
                current_episode_total_reward += reward

                if done_for_learning:
                    break # End episode if simulation is done or episode steps are completed

            episode_rewards.append(current_episode_total_reward)
            print(f"Episode {i_episode}/{NUM_EPISODES}, "
                  f"Total Reward: {current_episode_total_reward:.2f}, "
                  f"Epsilon: {epsilon:.3f}, "
                  f"Steps taken: {t_step_in_episode + 1}")

        torch.save(agent.policy_net.state_dict(), 'checkpoint.pth')

    except Exception as e:
        print(f"An error occurred during training: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Ensure the SUMO simulation is closed even if an error occurs
        env.close()
        print("Training complete.")

    return episode_rewards

if __name__ == '__main__':
    print("Starting SSE-DQN Training (SUMO Environment)...")

    # --- Basic Sanity Checks ---
    print(f"\nSTATE_SIZE is {STATE_SIZE}")
    print(f"ACTION_SIZE is {ACTION_SIZE}")
    print(f"COMPRESSED_VALUE_UNITS: {COMPRESSED_VALUE_UNITS}, GRU_UNITS: {GRU_UNITS}")
    if COMPRESSED_VALUE_UNITS != GRU_UNITS:
        print("WARNING: COMPRESSED_VALUE_UNITS and GRU_UNITS are not equal. "
              "The 'New Equation' division might require broadcasting or review if dimensions differ.")
    else:
        print("COMPRESSED_VALUE_UNITS and GRU_UNITS are equal, as expected for 'New Equation'.")

    # Test SSEN network instantiation and forward pass
    print("\nTesting SSEN network instantiation and forward pass...")
    try:
        dummy_state_batch = torch.rand(BATCH_SIZE, STATE_SIZE).to(DEVICE)
        # Ensure action_size is correct for the test model
        ssen_test_model = SSEN(STATE_SIZE, ACTION_SIZE).to(DEVICE)
        ssen_test_model.eval()
        with torch.no_grad():
            q_vals, h_next = ssen_test_model(dummy_state_batch)
        print(f"SSEN Q-values shape: {q_vals.shape} (Expected: ({BATCH_SIZE}, {ACTION_SIZE}))")
        print(f"SSEN next hidden state shape: {h_next.shape} (Expected: (1, {BATCH_SIZE}, {GRU_UNITS}))")

        dummy_state_single = torch.rand(1, STATE_SIZE).to(DEVICE)
        h_single_initial = torch.zeros(1, 1, GRU_UNITS).to(DEVICE)
        q_vals_single, h_next_single = ssen_test_model(dummy_state_single, h_single_initial)
        print(f"SSEN Q-values shape (single): {q_vals_single.shape} (Expected: (1, {ACTION_SIZE}))")
        print(f"SSEN next hidden state shape (single): {h_next_single.shape} (Expected: (1, 1, {GRU_UNITS}))")
        print("SSEN test pass successful.")
    except Exception as e:
        print(f"Error during SSEN test: {e}")
        import traceback
        traceback.print_exc()

    # Test Agent action selection
    print("\nTesting Agent action selection...")
    try:
        # Ensure action_size is correct for the test agent
        test_agent = SSE_DQNAgent(STATE_SIZE, ACTION_SIZE, DEVICE)
        dummy_env_state_np = np.random.rand(STATE_SIZE)
        initial_h_for_act = torch.zeros(1, 1, GRU_UNITS).to(DEVICE)
        action, next_h_for_act = test_agent.act(dummy_env_state_np, epsilon=0.1, h_0=initial_h_for_act)
        print(f"Agent selected action: {action} (Expected: 0 to {ACTION_SIZE-1})")
        print(f"Agent's next hidden state shape for acting: {next_h_for_act.shape} (Expected: (1, 1, {GRU_UNITS}))")
        print("Agent action selection test pass successful.")
    except Exception as e:
        print(f"Error during Agent action selection test: {e}")
        import traceback
        traceback.print_exc()

    # --- Run actual training ---
    # IMPORTANT: Replace 'path/to/your/sumo.cfg' with the actual path to your SUMO configuration file
    sumo_config_file = 'kulim_traffic_2.sumocfg' # <--- !!! REPLACE THIS LINE !!!

    if not os.path.exists(sumo_config_file):
        print(f"\nError: SUMO configuration file not found at '{sumo_config_file}'")
        print("Please update the 'sumo_config_file' variable with the correct path.")
    else:
        print(f"\nStarting actual training run using SUMO config: {sumo_config_file}")
        # The train_agent function now returns episode rewards
        training_rewards = train_agent(sumo_config_file)

        # You can optionally plot the training rewards here
        try:
            import matplotlib.pyplot as plt
            plt.plot(training_rewards)
            plt.xlabel("Episode")
            plt.ylabel("Total Reward per Episode")
            plt.title("SSE-DQN Training - SUMO Environment (Lane-based State & Penalized Reward)")
            plt.grid(True)
            plt.savefig("sse_dqn_sumo_training_rewards_lane_based_penalized.png")
            print("Training rewards plot saved to sse_dqn_sumo_training_rewards_lane_based_penalized.png")
            # plt.show() # Uncomment to display plot if in interactive environment
        except ImportError:
            print("\nmatplotlib not found, skipping plot generation.")
        except Exception as e:
            print(f"\nError generating plot: {e}")


    print("\nScript execution finished.")
