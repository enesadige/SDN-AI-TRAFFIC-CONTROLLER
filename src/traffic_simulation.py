from mininet.net import Mininet
from mininet.node import RemoteController, OVSKernelSwitch
from mininet.link import TCLink
from mininet.log import setLogLevel
from functools import partial
import time
from network_topology import get_all_topologies
import json
import sys
import os
import random

# ============================================================================
# SETTINGS & TRAFFIC POOL (LIGHTWEIGHT)
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SCENARIO_POOL = [
    # --- Düşük / Orta ---
    ("IoT_Sensor",     0.5,  1883, 'stable'),
    ("Web_HTTP",       3.0,  80,   'bursty'),
    ("DNS_Query",      0.8,  53,   'bursty'),
    ("VoIP_Call",      1.8,  5060, 'stable'),
    ("SSH_Tunnel",     0.6,  22,   'bursty'),
    
    # --- Orta / Yüksek ---
    ("Video_1080p",    14.0, 443,  'heavy_fluctuation'),
    ("Zoom_Meeting",   18.0, 8801, 'stable'),
    ("File_Upload",    25.0, 21,   'ramp'),
    ("Game_Update",    35.0, 80,   'ramp'),
    
    # --- Elephant (AMA ABARTI YOK) ---
    ("Netflix_4K",     55.0, 443,  'stable'),
    ("Database_Sync",  75.0, 1433, 'stable'),
    ("Steam_Download", 120.0,27015,'ramp'),
    ("Backup_Job",     160.0,2049, 'stable')
]

def generate_iperf_command(target_ip, speed, port, duration, profile_type):
    if profile_type == 'stable':
        actual_speed = speed * random.uniform(0.9, 1.05)
        return f"iperf -c {target_ip} -u -b {actual_speed:.2f}M -t {duration} -p {port}"

    elif profile_type == 'bursty':
        chunk = duration / 3
        cmds = []
        for _ in range(3):
            s = speed * random.choice([0.4, 1.2])
            cmds.append(f"iperf -c {target_ip} -u -b {s:.2f}M -t {chunk} -p {port}")
        return " ; ".join(cmds)

    elif profile_type == 'ramp':
        chunk = duration / 3
        cmds = []
        for i in range(3):
            s = speed * (0.7 + (i * 0.25))
            cmds.append(f"iperf -c {target_ip} -u -b {s:.2f}M -t {chunk} -p {port}")
        return " ; ".join(cmds)

    elif profile_type == 'heavy_fluctuation':
        chunk = duration / 4
        cmds = []
        for _ in range(4):
            s = speed * random.uniform(0.7, 1.4)
            cmds.append(f"iperf -c {target_ip} -u -b {s:.2f}M -t {chunk} -p {port}")
        return " ; ".join(cmds)

    return f"iperf -c {target_ip} -u -b {speed}M -t {duration} -p {port}"

def run_concurrent_test(net, h1, h2):
    print("\n PARALLEL TEST (LIGHT MODE)")
    print("   Goal: Model intervention exists but paths are not choked.")

    TOTAL_BATCHES = 7  # ⬅️ AZALTILDI

    for batch_id in range(TOTAL_BATCHES):
        num_flows = random.randint(2, 3)  # ⬅️ 4 yok
        selected_scenarios = random.sample(SCENARIO_POOL, num_flows)

        batch_duration = random.randint(12, 20)  # ⬅️ Daha kısa

        print(f"\n--- ROUND {batch_id+1}/{TOTAL_BATCHES} | {batch_duration}s ---")

        commands = []
        active_names = []

        for (name, speed, port, profile) in selected_scenarios:
            actual_port = port + (batch_id * 5)
            cmd_body = generate_iperf_command(
                h2.IP(), speed, actual_port, batch_duration, profile
            )
            final_cmd = f"({cmd_body}) > /dev/null 2>&1 &"
            commands.append(final_cmd)
            active_names.append(f"{name}({int(speed)}M)")

        print(f"    Active Flows: {', '.join(active_names)}")

        for cmd in commands:
            h1.cmd(cmd)

        for i in range(batch_duration):
            sys.stdout.write(f"\r    Traffic flowing... {batch_duration-i}s")
            sys.stdout.flush()
            time.sleep(1)

        print("\n    Round finished, cleaning up...")
        h1.cmd("killall -q iperf")
        time.sleep(2)

def run_topology_test(topo_class, topo_name, path_config, num_paths):
    print("\n" + "="*70)
    print(f"TOPO: {topo_name} | PATH: {num_paths}")
    print("="*70)

    config_path = os.path.join(BASE_DIR, "../config/topology_config.json")
    with open(config_path, 'w') as f:
        json.dump({
            "topology_name": topo_name,
            "path_config": path_config,
            "num_paths": num_paths
        }, f, indent=2)

    setLogLevel("info")

    topo = topo_class()
    net = Mininet(
        topo=topo,
        switch=partial(OVSKernelSwitch, protocols="OpenFlow13"),
        link=TCLink,
        controller=None
    )

    net.addController("c0", controller=RemoteController, ip="127.0.0.1", port=6654)
    net.start()

    print(" Network stabilizing (8s)...")
    time.sleep(8)

    h1, h2 = net.get("h1"), net.get("h2")

    try:
        run_concurrent_test(net, h1, h2)
    except KeyboardInterrupt:
        print("User stopped.")

    net.stop()
    h1.cmd("killall -q iperf")

def run():
    print(" SCENARIO RUNNER (LIGHT LOAD MODE)")
    for topo_class, topo_name, num_paths, path_config in get_all_topologies():
        run_topology_test(topo_class, topo_name, path_config, num_paths)

if __name__ == "__main__":
    if os.geteuid() != 0:
        print("Run with sudo.")
        sys.exit(1)
    run()
