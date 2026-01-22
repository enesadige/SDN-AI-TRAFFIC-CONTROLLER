from mininet.topo import Topo

# ============================================================================
# UPDATED TOPOLOGY FILE - REALISTIC NETWORKS FOR TESTING
# (30% EXTENDED VERSION: bw and capacity_mbps -> x1.3)
# ============================================================================

BW_SCALE = 1.3

def S(x):
    return int(round(x * BW_SCALE))


# ============================================================================
# TOPO 1: Realistic ISP Backbone (Multi-Hop Internet)
# ============================================================================
class TopoRealISP(Topo):
    def build(self):
        s1 = self.addSwitch('s1')  # Edge Router (Ev/Ofis çıkışı)
        s2 = self.addSwitch('s2')  # Data Center Gateway

        core1 = self.addSwitch('s3')  # Ana Omurga 1
        core2 = self.addSwitch('s4')  # Ana Omurga 2
        agg1  = self.addSwitch('s5')  # Toplayıcı 1
        agg2  = self.addSwitch('s6')  # Toplayıcı 2

        h1 = self.addHost('h1', ip='10.0.0.1/24')
        h2 = self.addHost('h2', ip='10.0.0.2/24')

        self.addLink(h1, s1, bw=1000, delay='0.1ms')
        self.addLink(s2, h2, bw=1000, delay='0.1ms')

        # --- PATH 1: Premium Fiber (Low Latency, Multi-Hop) ---
        # s1 -> agg1 -> core1 -> s2
        self.addLink(s1, agg1,    bw=S(400), delay='2ms',  jitter='0.5ms')  # 400 -> 520
        self.addLink(agg1, core1, bw=S(900), delay='1ms',  jitter='0.1ms')  # 900 -> 1170
        self.addLink(core1, s2,   bw=S(300), delay='2ms',  jitter='0.5ms')  # 300 -> 390 (bottleneck)

        # --- PATH 2: Standard ADSL/VDSL (Bottlenecked) ---
        self.addLink(s1, core2, bw=S(75), delay='15ms', jitter='4ms', max_queue_size=300)  # 75 -> 98 (bottleneck)
        self.addLink(core2, s2, bw=S(85), delay='15ms', jitter='4ms', max_queue_size=300)  # 85 -> 110

        # --- PATH 3: Backup Satellite/LTE (High Latency, Lossy) ---
        self.addLink(s1, agg2, bw=S(60), delay='40ms', jitter='15ms', loss=0.5)   # 60 -> 78
        self.addLink(agg2, s2, bw=S(60), delay='40ms', jitter='15ms', loss=0.5)   # 60 -> 78


# ============================================================================
# TOPO 2: Mesh Campus Network (Redundant and Complex)
# ============================================================================
class TopoCampusMesh(Topo):
    def build(self):
        s1 = self.addSwitch('s1')  # Fakülte Binası
        s2 = self.addSwitch('s2')  # Rektörlük Sunucusu

        sw_a = self.addSwitch('s3')
        sw_b = self.addSwitch('s4')
        sw_c = self.addSwitch('s5')
        sw_d = self.addSwitch('s6')

        h1 = self.addHost('h1', ip='10.0.0.1/24')
        h2 = self.addHost('h2', ip='10.0.0.2/24')

        self.addLink(h1, s1, bw=1000)
        self.addLink(s2, h2, bw=1000)

        # PATH 1: Short but Narrow (Direct Link)
        self.addLink(s1, sw_a, bw=S(200), delay='3ms')  # 200 -> 260
        self.addLink(sw_a, s2, bw=S(200), delay='3ms')  # 200 -> 260 (bottleneck)

        # PATH 2: Long but Broad (Via Backbone)
        self.addLink(s1, sw_b,   bw=S(400),  delay='2ms')  # 400 -> 520 (bottleneck)
        self.addLink(sw_b, sw_c, bw=S(1000), delay='1ms')  # 1000 -> 1300
        self.addLink(sw_c, s2,   bw=S(500),  delay='2ms')  # 500 -> 650

        # PATH 3: Legacy Infrastructure (Interference on Cross-link)
        self.addLink(s1, sw_d, bw=S(100), delay='10ms', loss=1)  # 100 -> 130
        self.addLink(sw_d, s2, bw=S(100), delay='10ms', loss=1)  # 100 -> 130 (bottleneck)


# ============================================================================
# TOPO 3: Cloud / Data Center (High Speed, Suitable for Burst Traffic)
# ============================================================================
class TopoCloudDC(Topo):
    def build(self):
        s1 = self.addSwitch('s1')  # Load Balancer
        s2 = self.addSwitch('s2')  # Backend

        path_nodes = []
        for i in range(4):
            sw = self.addSwitch(f's{i+3}')
            path_nodes.append(sw)

        h1 = self.addHost('h1', ip='10.0.0.1/24')
        h2 = self.addHost('h2', ip='10.0.0.2/24')

        self.addLink(h1, s1, bw=2000)
        self.addLink(s2, h2, bw=2000)

        caps = [1000, 800, 500, 300]  # -> x1.3
        delays = ['1ms', '1ms', '2ms', '5ms']

        for i, node in enumerate(path_nodes):
            self.addLink(s1, node, bw=S(caps[i]), delay=delays[i], jitter='0.05ms')
            self.addLink(node, s2, bw=S(caps[i]), delay=delays[i], jitter='0.05ms')


def get_all_topologies():
    return [
        (TopoRealISP, "Topo_RealISP_Test", 3, {
            # capacity_mbps: path bottleneck'e göre yazılmıştı -> x1.3
            1: {'capacity_mbps': 200.0 * BW_SCALE, 'delay_ms': 5.0},   # 200 -> 260
            2: {'capacity_mbps': 45.0  * BW_SCALE, 'delay_ms': 30.0},  # 45  -> 58.5
            3: {'capacity_mbps': 30.0  * BW_SCALE, 'delay_ms': 80.0}   # 30  -> 39
        }),
        (TopoCampusMesh, "Topo_Campus_Test", 3, {
            1: {'capacity_mbps': 100.0 * BW_SCALE, 'delay_ms': 6.0},   # 100 -> 130
            2: {'capacity_mbps': 300.0 * BW_SCALE, 'delay_ms': 5.0},   # 300 -> 390
            3: {'capacity_mbps': 50.0  * BW_SCALE, 'delay_ms': 20.0}   # 50  -> 65
        }),
        (TopoCloudDC, "Topo_CloudDC_Test", 4, {
            1: {'capacity_mbps': 800.0 * BW_SCALE, 'delay_ms': 2.0},   # 800 -> 1040
            2: {'capacity_mbps': 600.0 * BW_SCALE, 'delay_ms': 2.0},   # 600 -> 780
            3: {'capacity_mbps': 400.0 * BW_SCALE, 'delay_ms': 4.0},   # 400 -> 520
            4: {'capacity_mbps': 200.0 * BW_SCALE, 'delay_ms': 10.0}   # 200 -> 260
        })
    ]

