import os
import logging
import torch

from transformers import (
    TrainerCallback,
    TrainingArguments,
    TrainerState,
    TrainerControl,
)

PREFIX_CHECKPOINT_DIR = "step"

class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
        os.makedirs(checkpoint_folder, exist_ok=True)

        modules_to_save = []
        for module_name in args.trainable_params.split(","):
            if len(module_name.strip()) > 0:
                modules_to_save.append(module_name)

        # Save trainable parameters if exist
        if modules_to_save:
            state_dict = kwargs["model"].state_dict()
            to_save = {}
            for key, value in state_dict.items():
                if any(module_name in key for module_name in modules_to_save):
                    to_save[key.replace("base_model.model.", "")] = value
            torch.save(to_save, os.path.join(checkpoint_folder, "trainable_params.bin"))
            logging.info(f"Trainable parameters saved at: {checkpoint_folder}")

        # Save LoRA adapter weight
        kwargs["model"].save_pretrained(checkpoint_folder)
        logging.info(f"LoRA adapter weights saved at: {checkpoint_folder}")

        return control
