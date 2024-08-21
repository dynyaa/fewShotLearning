from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")




top_left_embeddings = np.load(r"C:\Users\robot\Desktop\embeddings\Top Left_embeddings.npy")
bottom_center_embeddings = np.load(r"C:\Users\robot\Desktop\embeddings\Bottom Center_embeddings.npy")
bottom_left_embeddings = np.load(r"C:\Users\robot\Desktop\embeddings\Bottom Left_embeddings.npy")
bottom_right_embeddings = np.load(r"C:\Users\robot\Desktop\embeddings\Bottom Right_embeddings.npy")
top_center_embeddings = np.load(r"C:\Users\robot\Desktop\embeddings\Top Center_embeddings.npy")
center_embeddings = np.load(r"C:\Users\robot\Desktop\embeddings\Center_embeddings.npy")
left_embeddings = np.load(r"C:\Users\robot\Desktop\embeddings\Left_embeddings.npy")
right_embeddings = np.load(r"C:\Users\robot\Desktop\embeddings\Right_embeddings.npy")
top_right_embeddings = np.load(r"C:\Users\robot\Desktop\embeddings\Top Right_embeddings.npy")

few_shot_prompt = f"""
Example 1:
Embedding: {top_left_embeddings[0][0].tolist()}  
Classification: Top Left

Example 2:
Embedding: {bottom_center_embeddings[0][0].tolist()}  
Classification: Bottom Center

Example 3:
Embedding: {bottom_left_embeddings[0][0].tolist()}  
Classification: Bottom Left

Example 4:
Embedding: {bottom_right_embeddings[0][0].tolist()}  
Classification: Bottom Right

Example 5:
Embedding: {top_center_embeddings[0][0].tolist()} 
Classification: Top Center

Example 6:
Embedding: {center_embeddings[0][0].tolist()} 
Classification: Center

Example 7:
Embedding: {left_embeddings[0][0].tolist()}  
Classification: Left

Example 8:
Embedding: {right_embeddings[0][0].tolist()}  
Classification: Right

Example 9:
Embedding: {top_right_embeddings[0][0].tolist()}  
Classification: Top Right
"""

embeddings_to_classify = [
    top_center_embeddings[2][0],
    bottom_left_embeddings[2][0],
    right_embeddings[2][0],
    left_embeddings[2][0],
    center_embeddings[2][0]
]

for i, embedding in enumerate(embeddings_to_classify):
    current_prompt = few_shot_prompt + f"""

Now classify the following embedding:
Embedding: {embedding.tolist()}
"""

    inputs = tokenizer(current_prompt, return_tensors="pt", truncation=True, max_length=1024)

    outputs = model.generate(**inputs, max_new_tokens=50)

    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Classification for embedding {i+1}: {response_text}")
