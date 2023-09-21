# Written by Yukang Chen
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import torch
import argparse

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--checkpoint_path', type=str, default="/dataset/models/checkpoint-1000")
    parser.add_argument('--trainable_params', type=str, default="embed,norm")
    args = parser.parse_args()
    return args


def main(args):
    path = args.checkpoint_path
    trainable_params = args.trainable_params.split(",")

    weights_all = torch.load(os.path.join(path, "pytorch_model.bin"))

    weights_trainable = {}
    weights_lora = {}
    for k in weights_all:
        if "lora" in k:
            k_new = k.replace("default.", "") if "default." in k else k
            weights_lora[k_new] = weights_all[k]
        else:
            if any([n in k for n in trainable_params]):
                weights_trainable[k[17:]] = weights_all[k]

    adapter_model = os.path.join(path, "adapter_model.bin")
    trainable_params = os.path.join(path, "trainable_params.bin")
    if not os.path.isfile(adapter_model):
        torch.save(weights_lora, adapter_model)
    torch.save(weights_trainable, trainable_params)

if __name__ == "__main__":
    args = parse_config()
    main(args)
