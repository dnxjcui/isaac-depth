from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from huggingface.modular_isaac import IsaacProcessor


def download_isaac_model():
    tokenizer = AutoTokenizer.from_pretrained("PerceptronAI/Isaac-0.1", trust_remote_code=True, use_fast=False)
    config = AutoConfig.from_pretrained("PerceptronAI/Isaac-0.1", trust_remote_code=True)
    processor = IsaacProcessor(tokenizer=tokenizer, config=config)
    model = AutoModelForCausalLM.from_pretrained("PerceptronAI/Isaac-0.1", trust_remote_code=True)
    # save the model locally
    model.save_pretrained("isaac_model")
    tokenizer.save_pretrained("isaac_tokenizer")


if __name__ == "__main__":
    download_isaac_model()