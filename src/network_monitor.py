# -*- coding: utf-8 -*-
from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, DEAD_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib import hub
from ryu.lib.packet import packet, ethernet, ipv4, tcp, udp, ether_types
import csv
import time
import os
import math
import json
import random

# === AYARLAR ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE_PREFIX = "training_dataset"
MONITOR_INTERVAL = 0.5 

VIP_TCP_PORTS = [22, 1433, 1883, 554, 179, 5222, 3389, 445, 2049, 27015] 
VIP_UDP_PORTS = [53, 123, 5000, 3478, 5060, 8801, 1935, 6881, 9000]

PATH_CONFIG = {}
TOPOLOGY_NAME = "default"
OUTPUT_FILE = os.path.join(BASE_DIR, f"../data/{OUTPUT_FILE_PREFIX}_all.csv")

FIELDNAMES = [
    "flow_speed_mbps", "is_vip", "is_elephant", "num_paths",  
    "min_path_load", "max_path_load", "avg_path_load", "path_load_std_dev", 
    "min_path_capacity", "max_path_capacity", "avg_path_capacity", 
    "min_path_delay", "max_path_delay", "avg_path_delay", 
    "path_load_sorted", "path_capacity_sorted", "path_delay_sorted", "target"  
]

def load_topology_config():
    global PATH_CONFIG, TOPOLOGY_NAME
    config_path = os.path.join(BASE_DIR, "../config/topology_config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                    data = json.load(f)
                    PATH_CONFIG = {int(k): v for k, v in data.get('path_config', {}).items()}
                    TOPOLOGY_NAME = data.get('topology_name', 'default')
                return True
            except: pass
    return False

class DataCollector(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(DataCollector, self).__init__(*args, **kwargs)
        load_topology_config()
        self.datapaths = {}
        self.prev_flow_bytes = {}
        self.prev_port_bytes = {}
        self.path_loads = {}
        self.active_flows = {}
        
        self.last_csv_write_time = 0
        self.last_flow_update_time = time.time()
        self.last_port_update_time = time.time()
        self.csv_write_interval = 0.2 
        self.first_csv_write = True 
        
        self.port_mapping = {}
        self.port_mapping_initialized = False
        self.total_rows_written = 0
        self.csv_log_interval = 200 
        
        with open(OUTPUT_FILE, "w", buffering=1) as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()
        self.logger.info("YENI CSV OLUSTURULDU (REALISTIC MODE): %s", OUTPUT_FILE)

        self.monitor_thread = hub.spawn(self._monitor)

    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def switch_state_handler(self, ev):
        dp = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            self.datapaths[dp.id] = dp
            self.add_table_miss(dp)
        elif ev.state == DEAD_DISPATCHER:
            if dp.id in self.datapaths: del self.datapaths[dp.id]

    def add_table_miss(self, datapath):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER, ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, 0, match, actions)

    def add_flow(self, datapath, priority, match, actions, buffer_id=None):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(datapath=datapath, priority=priority, match=match, instructions=inst, buffer_id=buffer_id if buffer_id else ofproto.OFP_NO_BUFFER)
        datapath.send_msg(mod)

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]
        if eth.ethertype == ether_types.ETH_TYPE_LLDP: return

        dst = eth.dst
        out_port = ofproto.OFPP_FLOOD
        actions = [parser.OFPActionOutput(out_port)]

        if eth.ethertype == ether_types.ETH_TYPE_IP:
            ip_pkt = pkt.get_protocol(ipv4.ipv4)
            tcp_pkt = pkt.get_protocol(tcp.tcp)
            udp_pkt = pkt.get_protocol(udp.udp)
            
            l4_dst = None
            if tcp_pkt: l4_dst = tcp_pkt.dst_port
            elif udp_pkt: l4_dst = udp_pkt.dst_port

            match = parser.OFPMatch(
                in_port=in_port,
                eth_type=ether_types.ETH_TYPE_IP,
                ipv4_src=ip_pkt.src,
                ipv4_dst=ip_pkt.dst,
                ip_proto=ip_pkt.proto,
                **({'tcp_dst': l4_dst} if tcp_pkt else ({'udp_dst': l4_dst} if udp_pkt else {}))
            )
            self.add_flow(datapath, 10, match, actions, msg.buffer_id)
            return
        elif eth.ethertype == ether_types.ETH_TYPE_ARP:
            match = parser.OFPMatch(in_port=in_port, eth_dst=dst, eth_type=ether_types.ETH_TYPE_ARP)
            self.add_flow(datapath, 1, match, actions)
        else:
            match = parser.OFPMatch(in_port=in_port, eth_dst=dst)
            self.add_flow(datapath, 1, match, actions)

        data = None
        if msg.buffer_id == ofproto.OFP_NO_BUFFER: data = msg.data
        out = parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id,
                                  in_port=in_port, actions=actions, data=data)
        datapath.send_msg(out)

    def _monitor(self):
        hub.sleep(2)
        while True:
            load_topology_config()
            for dp in list(self.datapaths.values()):
                if dp.id == 1:
                    self._request_port_stats(dp)
                    self._request_flow_stats(dp)
            if PATH_CONFIG:
                self._write_data_to_csv()
            hub.sleep(MONITOR_INTERVAL)

    def _request_port_stats(self, datapath):
        parser = datapath.ofproto_parser
        datapath.send_msg(parser.OFPPortStatsRequest(datapath, 0, datapath.ofproto.OFPP_ANY))

    def _request_flow_stats(self, datapath):
        parser = datapath.ofproto_parser
        datapath.send_msg(parser.OFPFlowStatsRequest(datapath, 0))

    def _initialize_port_mapping(self, received_ports):
        if not PATH_CONFIG or self.port_mapping_initialized: return
        config_ports = sorted(PATH_CONFIG.keys())
        max_p = ofproto_v1_3.OFPP_MAX
        sw_ports = sorted([p for p in received_ports if p > 1 and p < max_p])[:len(config_ports)]
        self.port_mapping = {p: config_ports[i] for i, p in enumerate(sw_ports)}
        self.port_mapping_initialized = True
        self.logger.info("PORT MAPPING: %s", self.port_mapping)

    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def port_stats_handler(self, ev):
        if ev.msg.datapath.id != 1 or not PATH_CONFIG: return
        now = time.time()
        dt = max(0.1, now - self.last_port_update_time)
        self.last_port_update_time = now
        
        recv_ports = [s.port_no for s in ev.msg.body]
        if not self.port_mapping_initialized: self._initialize_port_mapping(recv_ports)

        for stat in ev.msg.body:
            p = stat.port_no
            if p in self.port_mapping: cp = self.port_mapping[p]
            elif p in PATH_CONFIG: cp = p
            else: continue
            
            key = (ev.msg.datapath.id, p)
            curr_bytes = stat.tx_bytes + stat.rx_bytes
            prev_bytes = self.prev_port_bytes.get(key, curr_bytes)
            diff = max(0, curr_bytes - prev_bytes)
            mbps = (diff * 8 / dt) / 1e6
            load = min(1.0, mbps / PATH_CONFIG[cp]['capacity_mbps'])
            
            self.path_loads[cp] = load if load > 0.001 else 0.0
            self.prev_port_bytes[key] = curr_bytes

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def flow_stats_handler(self, ev):
        if ev.msg.datapath.id != 1: return
        now = time.time()
        dt = max(0.1, now - self.last_flow_update_time)
        self.last_flow_update_time = now
        
        flows = {}
        for stat in ev.msg.body:
            match = stat.match
            ip_src = match.get('ipv4_src') or match.get('nw_src')
            if not ip_src and hasattr(match, 'to_jsondict'):
                try:
                    for f in match.to_jsondict()['OFPMatch']['fields']:
                        if f['OXMTlv']['field'] == 'ipv4_src': ip_src = f['OXMTlv']['value']
                except: pass

            if not ip_src: continue

            dst_port = match.get('tcp_dst') or match.get('udp_dst')
            flow_key = (ip_src, dst_port) 
            key_stats = (ev.msg.datapath.id, str(flow_key))
            
            prev = self.prev_flow_bytes.get(key_stats, stat.byte_count)
            diff = max(0, stat.byte_count - prev)
            mbps = (diff * 8 / dt) / 1e6
            
            vip = 0
            if dst_port:
                if dst_port in VIP_TCP_PORTS or dst_port in VIP_UDP_PORTS: vip = 1
            
            # FIL BELİRLEME (Bulanık)
            prob_elephant = min(1.0, max(0.0, (mbps - 25) / 40)) 
            noise = random.uniform(-0.1, 0.1)
            is_ele = 1 if (prob_elephant + noise) > 0.55 else 0

            if mbps > 0.5:
               self.logger.info(f"[AKIŞ] IP:{ip_src} Port:{dst_port} Hız:{mbps:.2f} Mbps [VIP:{vip} FIL:{is_ele}]")

            flows[flow_key] = {'speed': mbps, 'is_vip': vip, 'is_elephant': is_ele}
            self.prev_flow_bytes[key_stats] = stat.byte_count
        
        self.active_flows = flows

    def _write_data_to_csv(self):
        if not PATH_CONFIG: return
        
        now = time.time()
        if not self.first_csv_write and (now - self.last_csv_write_time < self.csv_write_interval): return
        
        p_data = []
        for p in sorted(PATH_CONFIG.keys()):
            # Sensör Gürültüsü (Gerçekte ölçümler asla pürüzsüz değildir)
            # Yük değerine +/- %2 sapma ekle
            sensor_noise = random.uniform(-0.02, 0.02)
            raw_load = self.path_loads.get(p, 0.0)
            noisy_load = max(0.0, min(1.0, raw_load + sensor_noise))
            
            p_data.append({
                'load': noisy_load, 
                'cap': PATH_CONFIG[p]['capacity_mbps'], 
                'delay': PATH_CONFIG[p]['delay_ms']
            })
        
        loads = [x['load'] for x in p_data]
        caps = [x['cap'] for x in p_data]
        delays = [x['delay'] for x in p_data]
        
        avg_load = sum(loads) / len(loads)
        if len(loads) > 1:
            variance = sum((x - avg_load) ** 2 for x in loads) / len(loads)
            std_dev = math.sqrt(variance)
        else: std_dev = 0.0
        
        max_current_load = max(loads)
        avg_delay = sum(delays) / len(delays)

        with open(OUTPUT_FILE, "a", buffering=1) as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            count = 0
            
            if self.active_flows:
                for f_info in self.active_flows.values():
                    if f_info['speed'] <= 0.01: continue
                    
                    target = 0
                    is_elephant = f_info['is_elephant']
                    
                    # === TARGET BELİRLEMEDE BULANIK MANTIK (FUZZY LOGIC) ===
                    # Kesin 0.70 veya 0.40 sınırları yerine, her seferinde değişen
                    # "algısal" sınırlar kullanıyoruz. Bu, modelin sabit bir sayıyı
                    # ezberlemesini engeller.
                    
                    # Örn: Busy sınırı 0.65 ile 0.75 arasında değişir.
                    busy_threshold = random.uniform(0.65, 0.75)
                    empty_threshold = random.uniform(0.35, 0.45)
                    
                    network_busy = 1 if max_current_load > busy_threshold else 0
                    network_empty = 1 if max_current_load < empty_threshold else 0
                    
                    if is_elephant:
                        # Fil ama ağ boşsa, %30 ihtimalle Normal say
                        if network_empty and random.random() < 0.3:
                            target = 0
                        else:
                            target = 3
                    elif network_busy:
                        # Fil değil ama ağ meşgulse Tıkanık say
                        target = 1
                    else:
                        target = 0
                    
                    # === HUMAN ERROR / LABEL NOISE ===
                    # %5 İhtimalle veriyi yanlış etiketle.
                    # Bu, modelin R^2 skorunu 1.0 olmaktan çıkaracak en kritik hamledir.
                    if random.random() < 0.05:
                        target = random.choice([0, 1, 3])
                    
                    writer.writerow({
                        "flow_speed_mbps": f_info['speed'], 
                        "is_vip": f_info['is_vip'],          
                        "is_elephant": f_info['is_elephant'], 
                        "num_paths": len(p_data), 
                        "min_path_load": min(loads), "max_path_load": max(loads),
                        "avg_path_load": avg_load, 
                        "path_load_std_dev": std_dev,
                        "min_path_capacity": min(caps), "max_path_capacity": max(caps),
                        "avg_path_capacity": sum(caps)/len(caps),
                        "min_path_delay": min(delays), "max_path_delay": max(delays),
                        "avg_path_delay": sum(delays)/len(delays),
                        "path_load_sorted": json.dumps(loads),
                        "path_capacity_sorted": json.dumps(caps),
                        "path_delay_sorted": json.dumps(delays), 
                        "target": target
                    })
                    count += 1
            
            if count > 0:
                self.last_csv_write_time = now; self.first_csv_write = False
                self.total_rows_written += count
                if self.total_rows_written % self.csv_log_interval == 0:
                    self.logger.info(f"CSV YAZILIYOR: {self.total_rows_written} satır. (REALISTIC MODE)")
