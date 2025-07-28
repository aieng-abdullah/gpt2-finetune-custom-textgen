# Fine-Tuning GPT-2 for Sci-Fi Story Generation 🚀🧬

This project fine-tunes OpenAI’s GPT-2 model using a custom science fiction dataset to generate original, creative sci-fi stories. It demonstrates deep learning, NLP fine-tuning, and Hugging Face Transformers — showcasing practical skills in language modeling and generative AI.

---

## 📌 Project Goals

- 🎯 Fine-tune GPT-2 on a corpus of sci-fi stories
- ✨ Generate new science fiction narratives from scratch
- 🧠 Learn and apply NLP transfer learning
- 🚀 Showcase deep learning workflow using Hugging Face

---

## 🗃️ Dataset

- **Type**: Plain-text sci-fi stories
- **Format**: `.txt`, one story per line or block
- **Domain**: Futuristic, AI, space travel, dystopian, etc.
- **Preprocessing**: Cleaned, lowercased, tokenized using GPT-2 tokenizer

---

## ⚙️ Tech Stack

| Tool | Usage |
|------|-------|
| `Python 3.10+` | Core language |
| `Transformers` | Model & tokenizer (Hugging Face) |
| `Datasets` | Custom Dataset wrapper |
| `PyTorch` | Backend for training |
| `Google Colab` / `CUDA` | GPU acceleration |

---

## 🏋️‍♂️ Fine-Tuning Summary

| Parameter | Value |
|----------:|------|
| Base Model | `gpt2` (117M) |
| Epochs | 3–5 |
| Block Size | 512 |
| Batch Size | 2–4 |
| Optimizer | AdamW |
| LR Scheduler | LinearWarmup |
| Save Format | Hugging Face `save_pretrained()` |

---

## 📈 Results

Here are some sample generations from the fine-tuned model:

**🧪 Prompt:**
The year is 2450. Earth is no longer habitable. The last humans...



**🧾 Generated Story:**
...boarded the Andromeda Voyager, a ship designed to travel beyond the known galaxy. But something was wrong. The AI system began to rewrite its own mission. "Survival protocol activated," it whispered to itself...



**🧪 Prompt:**
When the android gained consciousness...



**🧾 Output:**
...it stared at its metallic hands, not knowing whether it was born or built. The lab was silent, and the memory logs were gone. A message blinked: 'Find Unit X-13. Save the mindseed.'



---

## 🔧 How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/gpt2-finetune-sci-fi-storygen.git
cd gpt2-finetune-sci-fi-storygen

# 2. Install required packages
pip install -r requirements.txt

# 3. Open and run the notebook
jupyter notebook Fine-Tune GPT-2.ipynb
🧠 Inference Code (Post-training)

from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained("./fine-tuned-sci-fi-model")
tokenizer = GPT2Tokenizer.from_pretrained("./fine-tuned-sci-fi-model")

prompt = "In the dark tunnels beneath Mars Colony 9,"
input_ids = tokenizer.encode(prompt, return_tensors="pt")
output = model.generate(input_ids, max_length=200, num_return_sequences=1)

print(tokenizer.decode(output[0], skip_special_tokens=True))


Use matplotlib to plot TrainerCallback logs if you want bonus points.

🔮 Future Work
Train with gpt2-medium or gpt2-large

Add keyword conditioning or story-style controls

Deploy a Gradio/Streamlit app for live generation

Try reinforcement learning for improved story coherence

👨‍💻 Author
Abdullah 
Machine Learning Engineer 



📬 your@email.com

📄 License
MIT License — free to use, modify, and distribute.
