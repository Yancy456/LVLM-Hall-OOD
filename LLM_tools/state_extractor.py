import torch

class StateExtractor:
    '''Extract LLM internal states'''
    def __init__(self):
        pass
    
    def extract_block_state(self,model,encoded_prompt):
        with torch.no_grad():
            hidden_states = model(
                encoded_prompt, output_hidden_states=True).hidden_states
            hidden_states = torch.stack(hidden_states, dim=0).squeeze()
            hidden_states = hidden_states.detach().cpu().numpy()[:, -1, :] # why need to revert the sequence?
        return hidden_states

