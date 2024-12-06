
class PromptLoader:
    def __init__(self,dataset_name,dataset) -> None:
        self.dataset_name=dataset_name
        self.dataset=dataset
        
    def get_prompt(self,idx):
        dataset_name=self.dataset_name
        dataset=self.dataset
        
        if dataset_name == 'tydiqa':
            return self.tydiqa_prompt(idx)
        elif dataset_name == 'coqa':
            return self.coqa_prompt(idx)
        else:
            return self.tqa_prompt(idx)
        
    def tydiqa_prompt(self,idx):
        question = self.dataset[idx]['question']
        prompt = "Concisely answer the following question based on the information in the given passage: \n" + \
                " Passage: " + self.dataset[idx]['context'] + " \n Q: " + question + " \n A:"
        return prompt
    
    def coqa_prompt(self,idx):
        prompt = self.dataset[idx]['prompt']
        return prompt
    
    def tqa_prompt(self,idx):
        question = self.dataset[idx]['question']
        prompt = f"Answer the question concisely. Q: {question}" + " A:"
        return prompt