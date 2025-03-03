import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"]='1'
from huggingface_hub import snapshot_download
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

class InferlessPythonModel:
    def initialize(self):
        model_id = "microsoft/Phi-3-mini-128k-instruct"
        snapshot_download(repo_id=model_id,allow_patterns=["*.safetensors"])
        self.model = AutoModelForCausalLM.from_pretrained(
            ,
            device_map="cuda",
            torch_dtype="auto",
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )

    def infer(self, input_data):
        prompt = input_data['prompt']
        roles = input_data['roles']

        generation_args = {
            "max_new_tokens": 500,
            "return_full_text": False,
            "temperature": 0.0,
            "do_sample": False,}
        messages = []
        messages.append({ "role": roles , "content" : prompt })
        output = self.pipe(messages, **generation_args)
        return {"result": output[0]['generated_text'] }

    def finalize(self):
        self.generator = None
        print("Pipeline finalized.", flush=True)
