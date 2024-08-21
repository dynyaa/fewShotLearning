import openai
import numpy as np

openai.api_key = 'api-key'

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
Embedding: {embedding.tolist()}.
"""

    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant who classifies embeddings based on given examples."},
            {"role": "user", "content": current_prompt}
        ]
    )

    classification = response['choices'][0]['message']['content'].strip()
    print(f"Classification for embedding {i+1}: {classification}")
