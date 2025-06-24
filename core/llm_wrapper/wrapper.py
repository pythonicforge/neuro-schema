import os
import yaml
from .quantized_llm import QuantizedLLM
from .transformer_llm import TransformerLLM
from scripts import logger

def load_llm():
    config_path = os.path.join(os.path.dirname(__file__), "model_config.yaml")
    try:
        logger.info("Reading model_config.yaml...")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)["model"]

        if config["type"] == "quantized":
            return QuantizedLLM(model_path=config["quant_path"])
        elif config["type"] == "transformer":
            return TransformerLLM(model_id=config["id"])
        else:
            raise ValueError("Invalid model type in config!")
    except Exception as e:
        logger.critical(e)