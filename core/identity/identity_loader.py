import os
import yaml
from scripts import logger

class IdentityMatrix:
    def __init__(self, path="core_identity.yaml"):
        self.path = os.path.join(os.path.dirname(__file__), path)

        with open(self.path, "r") as f:
            data = yaml.safe_load(f)["identity"]

        self.name = data["name"]
        self.version = data["version"]
        self.personality = data["personality"]
        self.values = data["values"]
        self.alignment_rules = data["alignment_rules"]
        self.boundaries = data["boundaries"]
        
    def describe_self(self):
        return f"I am {self.name}, driven by values like {', '.join(self.values[:3])}..."
    
    def should_respond(self, intent):
        for blocked in self.boundaries["avoid"]:
            if blocked in intent.lower():
                return False
        return True

