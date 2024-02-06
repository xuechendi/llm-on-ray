#!/bin/bash
#CONDA_HOME=$( cd -- "$( dirname -- "$( dirname -- "$(which conda)" )" )" &> /dev/null && pwd )
# pip install .[cpu,ui] -f https://developer.intel.com/ipex-whl-stable-cpu -f https://download.pytorch.org/whl/torch_stable.html
CONDA_HOME=/root/anaconda3
conda_env=llm-on-ray
ethinf=enp224s0f1
source $CONDA_HOME/bin/activate $conda_env
node_ip=$(ip addr show ${ethinf} | grep 'inet ' | awk '{print $2}' | cut -d'/' -f1)
export no_proxy="$node_ip, localhost, $no_proxy"

echo $no_proxy
user=root
#export LOG_LEVEL=info


echo python -u ui/start_ui.py --node_user_name $user --conda_env_name $conda_env --master_ip_port "$node_ip:6379"
python -u ui/start_ui.py --node_user_name $user --conda_env_name $conda_env --master_ip_port "$node_ip:6379"
