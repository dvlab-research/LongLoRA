## installation

conda create -n longlora python=3.9
conda activate longlora
# below command will install intel_extension_for_pytorch==2.0.110+xpu as default
# you can install specific ipex/torch version for your need
pip install --pre --upgrade bigdl-llm[xpu] -f https://developer.intel.com/ipex-whl-stable-xpu
pip install einops
pip install peft==0.5.0
pip install transformers==4.34.0
pip install accelerate==0.23.0
pip install oneccl_bind_pt==2.0.100 -f https://developer.intel.com/ipex-whl-stable-xpu
pip install git+https://github.com/microsoft/DeepSpeed.git@78c518e
pip install git+https://github.com/intel/intel-extension-for-deepspeed.git@ec33277

