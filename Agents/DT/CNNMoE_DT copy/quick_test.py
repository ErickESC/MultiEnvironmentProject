import torch
import torch.nn as nn
import numpy as np
import os
import sys
import json
from typing import Dict, List, Any, Tuple
import logging
from trainCNNMoE_v2 import MultiGameDecisionTransformer, GameConfig, DEVICE, unflatten_action

sys.path.append("C:/Research/AffectivelyFramework/")
sys.path.append("C:/Research/AffectivelyFramework/affectively/")
sys.path.append("C:/Research/")
import affectively
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiGameDTEvaluator:
    def __init__(self, model_path, game_configs_path):
        # Load game configs
        with open(game_configs_path, "r") as f:
            config_dict = json.load(f)
        
        self.game_configs = {}
        for name, config in config_dict.items():
            # Ensure action_space is loaded from config
            self.game_configs[name] = GameConfig(**config)
        
        # Create model with the same parameters as training
        hidden_size = 128
        n_layer = 3
        n_head = 4
        dropout = 0.1
        
        self.model = MultiGameDecisionTransformer(
            self.game_configs, hidden_size, n_layer, n_head, dropout, experts=True, num_of_features=10
        ).to(DEVICE)
        
        ## Load model weights
        #self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        #self.model.eval()
        #
        #logger.info(f"Loaded model from {model_path}")
    
    def quick_test(self):
        #games, states, actions, returns_to_go, timesteps, attention_mask=None
        print(self.game_configs)
        timesteps = [[i for i in range(63)]]

        print(self.model(["solid"],torch.rand(1,64, 3,250,240),torch.randint(0,3,(1,64,1)),torch.rand(1,64,1),torch.tensor(timesteps)))

def main():
    # Configuration
    model_path = "agents/game_obs/DT/MultiGame/Results/MG_DT_v2/best_model_MoE.pt"
    game_configs_path = "agents/game_obs/DT/MultiGame/Results/MultiGame_DT_v1/game_configs.json"
    
    # Initialize evaluator
    evaluator = MultiGameDTEvaluator(model_path, game_configs_path)
    
    # Example usage for Solid game
    try:
      
        evaluator.quick_test()
    
      
    except ImportError:
        logger.warning("Solid environment not available, skipping evaluation")

if __name__ == "__main__":
    main()