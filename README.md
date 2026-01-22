# SDN AI Traffic Controller

This project implements an intelligent traffic management system for Software-Defined Networks (SDN) using the Ryu controller and Deep Learning/Machine Learning techniques.

## 🚀 Features

*   **Hybrid AI Model:** Combines Dense Neural Networks (Scalar features) and Transformer/Attention mechanisms (Sequence features).
*   **Dynamic Traffic Rerouting:** Automatically reroutes "Congestion" causing flows to optimal paths.
*   **Elephant Flow Detection:** Identifies high-bandwidth flows.
*   **Real-time Monitoring:** Collects metrics via Prometheus & Grafana.

## 📂 Project Structure

```
GitHub_Ready_Project/
├── config/                 # Configuration files (Topology, Controller settings)
├── data/                   # Dataset storage (Raw CSVs, Merged, Cleaned)
├── models/                 # Trained AI models (.h5) and Scalers (.pkl)
├── src/                    # Source code
│   ├── sdn_traffic_controller.py  # Main Ryu Controller (formerly ryu_controller.py)
│   ├── network_monitor.py         # Data Collector (formerly data_collector.py)
│   ├── network_topology.py        # Mininet Topologies (formerly topo.py)
│   └── traffic_simulation.py      # Traffic Generator (formerly scenario_runner.py)
└── training/               # Model Training
    ├── model_trainer.py           # Training Script (formerly train_model.py)
    ├── data_preprocessor.py       # Data Cleaner (formerly data_cleaner.py)
    └── dataset_merger.py          # CSV Merger (formerly data_merging.py)
```

## 🛠️ Installation & Setup (Ubuntu/VM)

1.  **Activate Virtual Environment:**
    Navigate to your project folder and activate your venv (e.g., `myenv`):
    ```bash
    source myenv/bin/activate
    ```

2.  **Start Grafana (for Visualization):**
    ```bash
    sudo systemctl start grafana-server
    ```

## 🚦 Usage & Workflow

Open 2 separate terminals (and activate venv in both).

### Terminal 1: Start Ryu Controller
Run the controller on port 6654 (as configured in the project):
```bash
ryu-manager --ofp-tcp-listen-port 6654 src/sdn_traffic_controller.py
```

### Terminal 2: Start Traffic Simulation
Run the Mininet simulation (requires sudo):
```bash
sudo python3 src/traffic_simulation.py
```

---

## 🧠 Model Training Workflow

If you want to train the AI model from scratch with new data:

1.  **Merge CSV Files:**
    If you have multiple raw CSV files in `data/`, merge them first:
    ```bash
    cd training
    python dataset_merger.py
    ```
    *Output:* `data/dataset_all_merged.csv`

2.  **Clean & Preprocess Data:**
    Prepare the data for the model (removes dead flows, organizes columns):
    ```bash
    python data_preprocessor.py
    ```
    *Output:* `data/dataset_cleaned_final.csv`

3.  **Train the Model:**
    Train the Hybrid AI model using the cleaned data:
    ```bash
    python model_trainer.py
    ```
    *Output:* Saves `.h5` model and `.pkl` scaler to `models/` folder.

## 📝 Requirements

Python dependencies are listed in `requirements.txt`.

### System Dependencies
To verify installed versions of system tools (Mininet, Grafana, Prometheus), run:

```bash
# Mininet
mn --version

# Grafana
grafana-server -v

# Prometheus
prometheus --version
```

