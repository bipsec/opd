# Peer Spark (read over SSH on 2026-11-20; raw readings, not re-probed by qopd.utils.env)

- Link: this node `enP2p1s0f1np1` 192.168.100.10/24 <-> peer `enP2p1s0f1np1` 192.168.100.11/24, 200000 Mb/s (ethtool, local side), ping rtt ~0.54 ms
- Hostname: promaxgb10-4efc, arch aarch64
- GPU: `GPU 0: NVIDIA GB10 (UUID: GPU-9031c48f-427e-d81b-856f-07291f3b050d)`, compute_cap 12.1
- Driver 580.173.02, CUDA Version (nvidia-smi) 13.0 — local node has driver 580.159.03
- /proc/meminfo: MemTotal 127535252 kB, MemAvailable 59256236 kB
- GPU processes: pid 629616 `/home/biplab.sarkar/envs/train/bin/python`, 56243 MiB; power.draw 50.25 W at probe time
- Docker: could not read — `permission denied while trying to connect to the docker API` (user not in docker group on the peer)
- Disk (home): 3.6T total, 3.3T free
- `~/Project/opd` does not exist on the peer
