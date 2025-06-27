import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from scripts import logger
import subprocess
# from llama_cpp import quantize

class TransformerLLM:
    def __init__(self, model_id, local_dir="models/full") -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = model_id
        self.local_dir = os.path.join(local_dir, model_id.split("/")[-1])

        if not os.path.exists(self.local_dir) or not os.listdir(self.local_dir):
            logger.info(f"Downloading model: {self.model_id} [{self.local_dir}]")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
            os.makedirs(self.local_dir, exist_ok=True)
            self.tokenizer.save_pretrained(self.local_dir)
            self.model.save_pretrained(self.local_dir)
            logger.success("Download completed!")
        else:
            logger.info("Model found at local directory! Loading..")
            self.tokenizer = AutoTokenizer.from_pretrained(self.local_dir)
            self.model = AutoModelForCausalLM.from_pretrained(self.local_dir)

    def quantize_to_gguf(self, quant_type="f16", output_dir="models/quantized"):
        """
        Quantize the HF safetensors model to GGUF using llama_cpp.quantize API.
        quant_type options: 'q4_0', 'q4_k_m', 'q5_1', 'f16', etc.
        """
        gguf_name = f"{self.model_id.split('/')[-1]}-{quant_type}.gguf"
        output_path = os.path.join(output_dir, gguf_name)
        os.makedirs(output_dir, exist_ok=True)

        logger.info("Starting quantization process...")
        gguf_name = f"{self.model_id.split('/')[-1]}-{quant_type}.gguf"
        output_path = os.path.join(output_dir, gguf_name)
        os.makedirs(output_dir, exist_ok=True)
        cmd = [
            "python",
            "llama.cpp/convert_hf_to_gguf.py",
            self.local_dir,
            "--outfile",
            output_path,
            "--outtype",
            quant_type
        ]
        subprocess.run(cmd, check=True)
        logger.success(f"Quantized model saved at: {output_path}")