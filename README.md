# fewShotLearning
### README Overview

#### Task Description:
The objective of this project was to perform few-shot learning on video embeddings to classify them into different categories based on their spatial location in a grid. The project involves processing video frames, generating embeddings using OpenAI's CLIP model, and then using various large language models (LLMs) to classify these embeddings with minimal examples. 

#### How the Task Was Solved:
1. **Video Embedding Generation**: 
   - The videos were processed to extract individual frames, which were then converted to embeddings using OpenAI's CLIP model. These embeddings represent the spatial features of the frames in a format that LLMs can understand.
   - The `capturingEmbeddings.py` script handles this process by extracting frames from the videos, converting each frame into embeddings, and saving these embeddings for further processing.

2. **Few-Shot Learning with LLMs**:
   - The generated embeddings were used as inputs to various LLMs, including GPT-3.5, GPT-4, GPT-Neo, and LLaMA, to classify them into predefined categories (e.g., "Top Left", "Bottom Center", etc.). The LLMs were provided with a few examples of classified embeddings, and then asked to classify new embeddings based on these examples.
   - Each script (`gpt3.5.py`, `gpt4.py`, `gptneo.py`, and `llama.py`) follows a similar structure: it loads the embeddings, prepares a few-shot prompt, and queries the respective LLM to classify new embeddings.

3. **Model Comparison**:
   - The project also explores the performance and capabilities of different LLMs in handling the same task, providing insights into how different models perform on few-shot learning tasks.

