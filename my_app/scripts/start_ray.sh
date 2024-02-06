#!/bin/bash
# CONDA_HOME=$( cd -- "$( dirname -- "$( dirname -- "$(which conda)" )" )" &> /dev/null && pwd )

ethinf=enp224s0f1
source /root/anaconda3/bin/activate
conda activate llm-on-ray
node_ip=$(ip addr show ${ethinf} | grep 'inet ' | awk '{print $2}' | cut -d'/' -f1)
#conda env list
export no_proxy="$node_ip, localhost, $no_proxy"


RAY_SERVE_ENABLE_EXPERIMENTAL_STREAMING=1 ray start --head --node-ip-address $node_ip --dashboard-host 0.0.0.0 --resources='{"queue_hardware": 5}'

