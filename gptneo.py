from transformers import GPTNeoForCausalLM, GPT2Tokenizer
import numpy as np

model_name = "EleutherAI/gpt-neo-2.7B"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPTNeoForCausalLM.from_pretrained(model_name)

top_left_embeddings = np.load(r"C:\Users\robot\Desktop\embeddings\Top Left_embeddings.npy")
bottom_center_embeddings = np.load(r"C:\Users\robot\Desktop\embeddings\Bottom Center_embeddings.npy")

few_shot_prompt = f"""
Example 1:
Embedding: {top_left_embeddings[0][0].tolist()[:20]}  # Further truncated embedding
Classification: Top Left

Now classify the following embedding:
Embedding: {top_left_embeddings[2][0].tolist()[:20]}
"""

inputs = tokenizer(few_shot_prompt, return_tensors="pt", truncation=True, max_length=1024)  

outputs = model.generate(**inputs, max_new_tokens=50)

response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Model's classification: {response_text}")
