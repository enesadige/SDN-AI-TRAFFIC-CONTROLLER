# -*- coding: utf-8 -*-
from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, DEAD_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib import hub
from ryu.lib.packet import packet, ethernet, ipv4, tcp, udp, ether_types
import time
import os
import math
import json
import numpy as np
import pickle

from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
import eventlet
from eventlet import wsgi

# TensorFlow'u sadece hata mesajlarını bastıracak şekilde ayarla
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.models import load_model

# === AYARLAR ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../models/hybrid_traffic_model.h5")
SCALER_PATH = os.path.join(BASE_DIR, "../models/scaler.pkl")
MONITOR_INTERVAL = 1.0  # Tahmin sikligi (saniye)

VIP_TCP_PORTS = [22, 1433, 1883, 554, 179, 5222, 3389, 445, 2049, 27015]
VIP_UDP_PORTS = [53, 123, 5000, 3478, 5060, 8801, 1935, 6881, 9000]

PATH_CONFIG = {}
TOPOLOGY_NAME = "default"
NUM_PATHS = 0

# Modelin Target Çıktılarının Anlamları
CLASS_MAPPING = {0: "NORMAL (Target 0)", 1: "CONGESTION (Target 1)", 2: "ELEPHANT (Target 3)"}

# Modelin Sequence girişi için beklediği maksimum yol sayısı
MAX_PATHS_INPUT = 5

# === MODEL ON/OFF (AI toggle) ===
MODEL_ENABLED = True
CONTROLLER_CONFIG_PATH = os.path.join(BASE_DIR, "../config/controller_config.json")

# =========================
# PROMETHEUS METRICS
# =========================

# Aksiyon bazlı paket sayacı (PacketIn üzerinden artar)
PACKETS_TOTAL = Counter(
    "stc_packets_total",
    "SmartTrafficController tarafından işlenen toplam paket sayısı",
    ["action"],  # forward / reroute / drop
)

# ✅ YENİ: Modelin verdiği karar sayacı (PacketIn gelmese bile artar)
DECISIONS_TOTAL = Counter(
    "stc_decisions_total",
    "Modelin verdigi toplam karar sayisi (packet seviyesinden bagimsiz)",
    ["action"],  # forward / reroute / drop
)

# Reroute olduğunda hangi path seçildi sayacı (path_id label küçük kalıyor, iyi)
SELECTED_PATH_TOTAL = Counter(
    "stc_selected_path_total",
    "Seçilen path sayacı (reroute/forward için)",
    ["action", "path_id"],  # action=reroute/forward, path_id=1/2/3...
)

# Path load (0-1)
PATH_LOAD_GAUGE = Gauge(
    "stc_path_load",
    "Her path için normalize edilmiş anlık yük (0-1)",
    ["path_id"],
)

# Path kapasite ve delay (config’ten sabit)
PATH_CAPACITY_Mbps = Gauge(
    "stc_path_capacity_mbps",
    "Her path için kapasite (Mbps) (config)",
    ["path_id"],
)
PATH_DELAY_MS = Gauge(
    "stc_path_delay_ms",
    "Her path için gecikme (ms) (config)",
    ["path_id"],
)

# Utilization (%) = load*100
PATH_UTIL_PCT = Gauge(
    "stc_path_utilization_pct",
    "Her path için anlık doluluk (%)",
    ["path_id"],
)

# Port throughput (rx/tx)
PORT_RX_Mbps = Gauge(
    "stc_port_rx_mbps",
    "Switch port RX throughput (Mbps)",
    ["dpid", "port_no", "path_id"],
)
PORT_TX_Mbps = Gauge(
    "stc_port_tx_mbps",
    "Switch port TX throughput (Mbps)",
    ["dpid", "port_no", "path_id"],
)

# Port drop/error sayaçları (cumulative)
PORT_RX_DROPPED = Gauge(
    "stc_port_rx_dropped_total",
    "Switch port rx_dropped (cumulative)",
    ["dpid", "port_no", "path_id"],
)
PORT_TX_DROPPED = Gauge(
    "stc_port_tx_dropped_total",
    "Switch port tx_dropped (cumulative)",
    ["dpid", "port_no", "path_id"],
)
PORT_RX_ERRORS = Gauge(
    "stc_port_rx_errors_total",
    "Switch port rx_errors (cumulative)",
    ["dpid", "port_no", "path_id"],
)
PORT_TX_ERRORS = Gauge(
    "stc_port_tx_errors_total",
    "Switch port tx_errors (cumulative)",
    ["dpid", "port_no", "path_id"],
)

# Aktif flow sayısı
ACTIVE_FLOWS_GAUGE = Gauge(
    "stc_active_flows",
    "FlowStats ile tespit edilen aktif flow sayısı",
)

# Model durumu
MODEL_ENABLED_GAUGE = Gauge(
    "stc_model_enabled",
    "Model açık mı? 1/0",
)

# AI inference süresi
AI_INFERENCE_SECONDS = Histogram(
    "stc_ai_inference_seconds",
    "AI model inference süresi (seconds)",
    buckets=(0.001, 0.003, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0),
)

# Packet-in handler processing süresi (Ryu karar verme + kural yazma)
PACKETIN_PROCESS_SECONDS = Histogram(
    "stc_packetin_process_seconds",
    "PacketIn işleme süresi (seconds)",
    buckets=(0.0005, 0.001, 0.003, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2),
)


def load_topology_config():
    global PATH_CONFIG, TOPOLOGY_NAME, NUM_PATHS
    config_path = os.path.join(BASE_DIR, "../config/topology_config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                    data = json.load(f)
                    PATH_CONFIG = {int(k): v for k, v in data.get('path_config', {}).items()}
                    TOPOLOGY_NAME = data.get('topology_name', 'default')
                    try:
                        NUM_PATHS = int(data.get('num_paths', len(PATH_CONFIG)))
                    except Exception:
                        NUM_PATHS = len(PATH_CONFIG)
                return True
            except Exception:
                pass
    return False


def load_controller_config():
    """controller_config.json içinden model_enabled flag'ini okumak için."""
    global MODEL_ENABLED
    if not os.path.exists(CONTROLLER_CONFIG_PATH):
        return

    try:
        with open(CONTROLLER_CONFIG_PATH, 'r') as f:
            data = json.load(f)
            if isinstance(data.get("model_enabled"), bool):
                MODEL_ENABLED = data["model_enabled"]
    except Exception:
        pass


def prometheus_app(environ, start_response):
    """Basit /metrics endpoint'i (Prometheus için)."""
    path = environ.get('PATH_INFO', '')
    if path == '/metrics':
        data = generate_latest()
        status = '200 OK'
        headers = [('Content-type', CONTENT_TYPE_LATEST)]
        start_response(status, headers)
        return [data]
    else:
        status = '404 Not Found'
        start_response(status, [('Content-type', 'text/plain')])
        return [b'Not Found']


class SmartTrafficController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(SmartTrafficController, self).__init__(*args, **kwargs)

        # --- Prometheus metrics HTTP server ---
        try:
            self.metrics_thread = hub.spawn(self._start_metrics_server)
            self.logger.info("Prometheus metrics endpoint: http://0.0.0.0:8001/metrics")
        except Exception as e:
            self.logger.error("Metrics server başlatılamadı: %s", e)

        load_topology_config()
        load_controller_config()

        self.datapaths = {}
        self.prev_flow_bytes = {}

        # port throughput hesaplamak için ayrı önceki değerler
        self.prev_port_rx_bytes = {}
        self.prev_port_tx_bytes = {}
        self.last_port_time = time.time()

        self.path_loads = {}
        self.active_flows = {}

        self.last_update_time = time.time()
        self.port_mapping = {}
        self.port_mapping_initialized = False
        self.reverse_port_mapping = {}

        # Akış -> AI kararının cache'i (konvoy mantığı için)
        self.flow_decisions = {}
        self.decision_ttl = 5.0
        self.decision_use_limit = 20

        # --- YAPAY ZEKA MODELİNİ YÜKLE ---
        self.ai_ready = False
        if MODEL_ENABLED:
            self.logger.info(" MODEL YUKLENIYOR: %s", MODEL_PATH)
            try:
                self.model = load_model(MODEL_PATH)
                with open(SCALER_PATH, 'rb') as f:
                    self.scaler = pickle.load(f)
                self.logger.info(" MODEL VE SCALER BASARIYLA YUKLENDI!")
                self.ai_ready = True
            except Exception as e:
                self.logger.error(" MODEL YUKLENEMEDI: %s", e)
                self.ai_ready = False
        else:
            self.logger.info(" MODEL DEVRE DISI (MODEL_ENABLED=False) - Sadece izleme modunda calisacak.")

        # config’ten sabit metric’leri bir kere bas
        self._publish_static_path_metrics()

        # Topoloji değişimini takip etmek için imza (signature)
        self._topo_signature = (TOPOLOGY_NAME, tuple(sorted(PATH_CONFIG.keys())), int(NUM_PATHS))

        self.monitor_thread = hub.spawn(self._monitor)

    def _start_metrics_server(self):
        listener = eventlet.listen(('0.0.0.0', 8001))
        wsgi.server(listener, prometheus_app)


    def _reset_topology_dependent_state(self, reason=""):
        # Topoloji değişince port↔path mapping ve port byte cache resetlenmeli
        self.port_mapping = {}
        self.reverse_port_mapping = {}
        self.port_mapping_initialized = False

        # Eski byte cache kalırsa dt/diff hatalı çıkar ve load %100 sapıtabilir
        self.prev_port_rx_bytes = {}
        self.prev_port_tx_bytes = {}

        # Eski path load’ları da temizleyelim
        self.path_loads = {}

        self.logger.info(" Topology change -> reset port/path mapping. %s", reason)

    def _publish_static_path_metrics(self):
        """PATH_CONFIG içindeki delay/capacity gibi sabitleri Prometheus’a yaz."""
        if not PATH_CONFIG:
            return
        for pid, info in PATH_CONFIG.items():
            try:
                PATH_CAPACITY_Mbps.labels(path_id=str(pid)).set(float(info.get("capacity_mbps", 0.0)))
                PATH_DELAY_MS.labels(path_id=str(pid)).set(float(info.get("delay_ms", 0.0)))
            except Exception:
                pass

    # ==========================================================
    # RYU DURUM / FLOW YÖNETİMİ
    # ==========================================================
    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def switch_state_handler(self, ev):
        dp = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            self.datapaths[dp.id] = dp
            self.add_table_miss(dp)
        elif ev.state == DEAD_DISPATCHER:
            if dp.id in self.datapaths:
                del self.datapaths[dp.id]

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
        mod = parser.OFPFlowMod(
            datapath=datapath,
            priority=priority,
            match=match,
            instructions=inst,
            buffer_id=buffer_id if buffer_id is not None else ofproto.OFP_NO_BUFFER,
        )
        datapath.send_msg(mod)

    # ==========================================================
    # PORT / FLOW STATS REQUEST
    # ==========================================================
    def _request_port_stats(self, datapath):
        parser = datapath.ofproto_parser
        datapath.send_msg(parser.OFPPortStatsRequest(datapath, 0, datapath.ofproto.OFPP_ANY))

    def _request_flow_stats(self, datapath):
        parser = datapath.ofproto_parser
        datapath.send_msg(parser.OFPFlowStatsRequest(datapath, 0))

    def _initialize_port_mapping(self, received_ports):
        if not PATH_CONFIG or self.port_mapping_initialized:
            return
        config_ports = sorted(PATH_CONFIG.keys())
        sw_ports = sorted([p for p in received_ports if p > 1 and p < ofproto_v1_3.OFPP_MAX])[: len(config_ports)]

        # switch_port -> path_id
        self.port_mapping = {p: config_ports[i] for i, p in enumerate(sw_ports)}
        # path_id -> switch_port
        self.reverse_port_mapping = {path_id: port for port, path_id in self.port_mapping.items()}
        self.port_mapping_initialized = True

        self.logger.info("PORT MAPPING: %s", self.port_mapping)
        self.logger.info("REVERSE PORT MAPPING: %s", self.reverse_port_mapping)

    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def port_stats_handler(self, ev):
        if ev.msg.datapath.id != 1 or not PATH_CONFIG:
            return

        recv_ports = [s.port_no for s in ev.msg.body]
        # PATH_CONFIG değiştiyse (topoloji değişimi), mapping’i yeniden kur
        if self.port_mapping_initialized and len(self.reverse_port_mapping) != len(PATH_CONFIG):
            self._reset_topology_dependent_state(reason="PATH_CONFIG size changed -> reinit mapping")

        if not self.port_mapping_initialized:
            self._initialize_port_mapping(recv_ports)

        now = time.time()
        dt = max(0.1, now - self.last_port_time)
        self.last_port_time = now

        dpid = str(ev.msg.datapath.id)

        for stat in ev.msg.body:
            p = stat.port_no

            # path_id eşle
            if p in self.port_mapping:
                path_id = self.port_mapping[p]
            elif p in PATH_CONFIG:
                path_id = p
            else:
                path_id = 0  # mapping dışı port

            path_id_s = str(path_id)
            port_s = str(p)

            # RX/TX throughput hesapla
            key = (dpid, port_s)

            curr_rx = getattr(stat, "rx_bytes", 0)
            curr_tx = getattr(stat, "tx_bytes", 0)

            prev_rx = self.prev_port_rx_bytes.get(key, curr_rx)
            prev_tx = self.prev_port_tx_bytes.get(key, curr_tx)

            diff_rx = max(0, curr_rx - prev_rx)
            diff_tx = max(0, curr_tx - prev_tx)

            rx_mbps = (diff_rx * 8 / dt) / 1e6
            tx_mbps = (diff_tx * 8 / dt) / 1e6

            self.prev_port_rx_bytes[key] = curr_rx
            self.prev_port_tx_bytes[key] = curr_tx

            # PATH LOAD hesapla (toplam trafik / kapasite)
            capacity = float(PATH_CONFIG.get(path_id, {}).get("capacity_mbps", 0.0)) if path_id in PATH_CONFIG else 0.0
            if capacity > 0:
                total_mbps = rx_mbps + tx_mbps
                load = min(1.0, total_mbps / capacity)
                self.path_loads[path_id] = load

                try:
                    PATH_LOAD_GAUGE.labels(path_id=path_id_s).set(load)
                    PATH_UTIL_PCT.labels(path_id=path_id_s).set(load * 100.0)
                except Exception:
                    pass

            try:
                PORT_RX_Mbps.labels(dpid=dpid, port_no=port_s, path_id=path_id_s).set(rx_mbps)
                PORT_TX_Mbps.labels(dpid=dpid, port_no=port_s, path_id=path_id_s).set(tx_mbps)

                PORT_RX_DROPPED.labels(dpid=dpid, port_no=port_s, path_id=path_id_s).set(getattr(stat, "rx_dropped", 0))
                PORT_TX_DROPPED.labels(dpid=dpid, port_no=port_s, path_id=path_id_s).set(getattr(stat, "tx_dropped", 0))
                PORT_RX_ERRORS.labels(dpid=dpid, port_no=port_s, path_id=path_id_s).set(getattr(stat, "rx_errors", 0))
                PORT_TX_ERRORS.labels(dpid=dpid, port_no=port_s, path_id=path_id_s).set(getattr(stat, "tx_errors", 0))
            except Exception:
                pass

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def flow_stats_handler(self, ev):
        if ev.msg.datapath.id != 1:
            return

        flows = {}
        for stat in ev.msg.body:
            match = stat.match

            if 'ipv4_src' not in match and 'nw_src' not in match:
                has_ip = False
                if hasattr(match, 'to_jsondict'):
                    try:
                        for f in match.to_jsondict()['OFPMatch']['fields']:
                            if f['OXMTlv']['field'] == 'ipv4_src':
                                has_ip = True
                    except Exception:
                        pass
                if not has_ip:
                    continue

            ip_src = match.get('ipv4_src') or match.get('nw_src')
            dst_port = match.get('tcp_dst') or match.get('udp_dst')

            key_stats = (ev.msg.datapath.id, str((ip_src, dst_port)))
            prev = self.prev_flow_bytes.get(key_stats, stat.byte_count)
            diff = max(0, stat.byte_count - prev)
            mbps = (diff * 8 / MONITOR_INTERVAL) / 1e6

            vip = 0
            if dst_port and (dst_port in VIP_TCP_PORTS or dst_port in VIP_UDP_PORTS):
                vip = 1

            if mbps > 0.1:
                flows[(ip_src, dst_port)] = {'speed': mbps, 'is_vip': vip}

            self.prev_flow_bytes[key_stats] = stat.byte_count

        self.active_flows = flows
        try:
            ACTIVE_FLOWS_GAUGE.set(len(flows))
        except Exception:
            pass

    # ==========================================================
    # AI TAHMİN + KARAR CACHE'İ
    # ==========================================================
    def _get_flow_key(self, ip_src, dst_port):
        safe_port = dst_port if dst_port is not None else 0
        return (ip_src, safe_port)

    def _get_output_port_for_path(self, path_id):
        if not path_id:
            return None
        if self.reverse_port_mapping:
            return self.reverse_port_mapping.get(path_id)
        for port, cp in self.port_mapping.items():
            if cp == path_id:
                return port
        return None

    def _select_best_path_for_flow(self, f_info, loads, caps, delays):
        if not PATH_CONFIG:
            return None
        if not loads or not caps or not delays:
            return None

        speed = f_info.get('speed', 0.0)
        is_vip = f_info.get('is_vip', 0)

        is_heavy = 1 if speed >= 20.0 else 0

        max_cap = max(caps) if caps else 1.0
        max_delay = max(delays) if delays else 1.0

        best_score = None
        best_path_id = None

        for idx, path_id in enumerate(sorted(PATH_CONFIG.keys())):
            if idx >= len(loads):
                break

            load = loads[idx]
            cap = caps[idx]
            delay = delays[idx]

            load_norm = load
            cap_norm = cap / max_cap
            delay_norm = delay / max_delay

            w_load = 0.4 + 0.3 * is_heavy
            w_delay = 0.4 + 0.3 * is_vip
            w_cap = 0.4 + 0.3 * is_heavy

            score = w_load * load_norm + w_delay * delay_norm - w_cap * cap_norm

            if best_score is None or score < best_score:
                best_score = score
                best_path_id = path_id

        return best_path_id

    def _predict_traffic(self):
        if not self.active_flows or not PATH_CONFIG:
            return

        p_data = []
        for p in sorted(PATH_CONFIG.keys()):
            raw_load = self.path_loads.get(p, 0.0)
            p_data.append(
                {
                    'load': raw_load,
                    'cap': PATH_CONFIG[p]['capacity_mbps'],
                    'delay': PATH_CONFIG[p]['delay_ms'],
                }
            )

        loads = [x['load'] for x in p_data]
        caps = [x['cap'] for x in p_data]
        delays = [x['delay'] for x in p_data]

        avg_load = sum(loads) / len(loads) if loads else 0.0
        min_load, max_load = (min(loads), max(loads)) if loads else (0.0, 0.0)

        if len(loads) > 1:
            variance = sum((x - avg_load) ** 2 for x in loads) / len(loads)
            std_dev = math.sqrt(variance)
        else:
            std_dev = 0.0

        seq_input = np.zeros((1, MAX_PATHS_INPUT, 3), dtype=np.float32)
        curr_paths = len(loads)
        for j in range(min(curr_paths, MAX_PATHS_INPUT)):
            seq_input[0, j, 0] = loads[j]
            seq_input[0, j, 1] = caps[j] / 1000.0
            seq_input[0, j, 2] = delays[j] / 100.0

        now = time.time()

        for flow_key, f_info in self.active_flows.items():
            ip_src, dst_port = flow_key

            raw_scalar = np.array(
                [[
                    f_info['speed'],
                    f_info['is_vip'],
                    len(p_data),
                    min_load,
                    max_load,
                    avg_load,
                    std_dev,
                    min(caps) if caps else 0.0,
                    max(caps) if caps else 0.0,
                    (sum(caps) / len(caps)) if caps else 0.0,
                    min(delays) if delays else 0.0,
                    max(delays) if delays else 0.0,
                    (sum(delays) / len(delays)) if delays else 0.0,
                ]]
            )

            scaled_scalar = self.scaler.transform(raw_scalar)

            t0 = time.time()
            prediction_probs = self.model.predict([scaled_scalar, seq_input], verbose=0)
            AI_INFERENCE_SECONDS.observe(time.time() - t0)

            predicted_class_idx = int(np.argmax(prediction_probs))
            confidence = float(np.max(prediction_probs))

            # ✅ YENİ: Karar sayacı (tahmin anında artar)
            try:
                if predicted_class_idx == 0:
                    DECISIONS_TOTAL.labels(action="forward").inc()
                elif predicted_class_idx == 1:
                    DECISIONS_TOTAL.labels(action="reroute").inc()
                elif predicted_class_idx == 2:
                    DECISIONS_TOTAL.labels(action="drop").inc()
                else:
                    DECISIONS_TOTAL.labels(action="unknown").inc()
            except Exception:
                pass

            chosen_path = None
            if predicted_class_idx == 1:
                chosen_path = self._select_best_path_for_flow(f_info, loads, caps, delays)

            # cache'e yaz
            self.flow_decisions[self._get_flow_key(ip_src, dst_port)] = {
                'class_idx': predicted_class_idx,
                'confidence': confidence,
                'timestamp': now,
                'use_count': 0,
                'path_id': chosen_path
            }

    # ==========================================================
    # PACKET_IN: AI kararını uygula
    # ==========================================================
    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        start = time.time()
        try:
            msg = ev.msg
            datapath = msg.datapath
            ofproto = datapath.ofproto
            parser = datapath.ofproto_parser
            in_port = msg.match['in_port']

            pkt = packet.Packet(msg.data)
            eth = pkt.get_protocols(ethernet.ethernet)[0]
            if eth.ethertype == ether_types.ETH_TYPE_LLDP:
                return

            dst = eth.dst
            default_actions = [parser.OFPActionOutput(ofproto.OFPP_FLOOD)]

            # Model kapalıysa klasik davran
            if not MODEL_ENABLED or not self.ai_ready:
                match = parser.OFPMatch(in_port=in_port, eth_dst=dst)
                self.add_flow(datapath, 1, match, default_actions)

                try:
                    PACKETS_TOTAL.labels(action="forward").inc()
                except Exception:
                    pass

                data = msg.data if msg.buffer_id == ofproto.OFP_NO_BUFFER else None
                out = parser.OFPPacketOut(
                    datapath=datapath,
                    buffer_id=msg.buffer_id,
                    in_port=in_port,
                    actions=default_actions,
                    data=data,
                )
                datapath.send_msg(out)
                return

            # === IP trafikte modele göre aksiyon ===
            if eth.ethertype == ether_types.ETH_TYPE_IP:
                ip_pkt = pkt.get_protocol(ipv4.ipv4)
                tcp_pkt = pkt.get_protocol(tcp.tcp)
                udp_pkt = pkt.get_protocol(udp.udp)

                l4_dst = None
                if tcp_pkt:
                    l4_dst = tcp_pkt.dst_port
                elif udp_pkt:
                    l4_dst = udp_pkt.dst_port

                match_kwargs = {
                    'in_port': in_port,
                    'eth_type': ether_types.ETH_TYPE_IP,
                    'ipv4_src': ip_pkt.src,
                    'ipv4_dst': ip_pkt.dst,
                    'ip_proto': ip_pkt.proto,
                }
                if tcp_pkt:
                    match_kwargs['tcp_dst'] = l4_dst
                elif udp_pkt:
                    match_kwargs['udp_dst'] = l4_dst
                match = parser.OFPMatch(**match_kwargs)

                flow_key = self._get_flow_key(ip_pkt.src, l4_dst)
                decision = self.flow_decisions.get(flow_key)

                actions = list(default_actions)
                priority = 10
                action_label = "forward"
                chosen_path_id = None

                if decision:
                    now = time.time()
                    if (now - decision.get('timestamp', 0) <= self.decision_ttl) and (
                        decision.get('use_count', 0) < self.decision_use_limit
                    ):
                        class_idx = decision.get('class_idx', 0)
                        decision['use_count'] = decision.get('use_count', 0) + 1
                        self.flow_decisions[flow_key] = decision

                        if class_idx == 0:
                            action_label = "forward"

                        elif class_idx == 1:
                            path_id = decision.get('path_id')
                            out_port = self._get_output_port_for_path(path_id) if path_id else None

                            if not out_port:
                                best_path_id = None
                                if self.path_loads:
                                    best_path_id = min(self.path_loads, key=self.path_loads.get)
                                out_port = self._get_output_port_for_path(best_path_id)
                                path_id = best_path_id

                            if out_port:
                                actions = [parser.OFPActionOutput(out_port)]
                                action_label = "reroute"
                                chosen_path_id = path_id
                            else:
                                action_label = "forward"

                        elif class_idx == 2:
                            actions = []
                            priority = 20
                            action_label = "drop"

                self.add_flow(datapath, priority, match, actions, msg.buffer_id)

                try:
                    PACKETS_TOTAL.labels(action=action_label).inc()
                except Exception:
                    pass

                if action_label in ("reroute", "forward"):
                    pid = str(chosen_path_id) if chosen_path_id else "none"
                    try:
                        SELECTED_PATH_TOTAL.labels(action=action_label, path_id=pid).inc()
                    except Exception:
                        pass

                if actions:
                    if msg.buffer_id == ofproto.OFP_NO_BUFFER:
                        out = parser.OFPPacketOut(
                            datapath=datapath,
                            buffer_id=ofproto.OFP_NO_BUFFER,
                            in_port=in_port,
                            actions=actions,
                            data=msg.data,
                        )
                        datapath.send_msg(out)

                return

            match = parser.OFPMatch(in_port=in_port, eth_dst=dst)
            self.add_flow(datapath, 1, match, default_actions)

            try:
                PACKETS_TOTAL.labels(action="forward").inc()
            except Exception:
                pass

            data = msg.data if msg.buffer_id == ofproto.OFP_NO_BUFFER else None
            out = parser.OFPPacketOut(
                datapath=datapath,
                buffer_id=msg.buffer_id,
                in_port=in_port,
                actions=default_actions,
                data=data,
            )
            datapath.send_msg(out)

        finally:
            PACKETIN_PROCESS_SECONDS.observe(time.time() - start)

    # ==========================================================
    # MONITOR THREAD: Port & Flow istatistikleri + AI tahmini
    # ==========================================================
    def _monitor(self):
        hub.sleep(2)
        while True:
            load_controller_config()
            load_topology_config()

            # Topoloji değiştiyse mapping resetle
            new_sig = (TOPOLOGY_NAME, tuple(sorted(PATH_CONFIG.keys())), int(NUM_PATHS))
            if getattr(self, '_topo_signature', None) != new_sig:
                old = getattr(self, '_topo_signature', None)
                self._topo_signature = new_sig
                self._reset_topology_dependent_state(reason=f"Topology signature changed {old} -> {new_sig}")

            try:
                MODEL_ENABLED_GAUGE.set(1 if MODEL_ENABLED else 0)
            except Exception:
                pass

            self._publish_static_path_metrics()

            for dp in list(self.datapaths.values()):
                if dp.id == 1:
                    self._request_port_stats(dp)
                    self._request_flow_stats(dp)

            if MODEL_ENABLED and self.ai_ready and PATH_CONFIG:
                self._predict_traffic()
            else:
                self.flow_decisions.clear()

            hub.sleep(MONITOR_INTERVAL)

