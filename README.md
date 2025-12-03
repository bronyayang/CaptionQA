# CaptionQA: Is Your Caption as Useful as the Image Itself?

> A utility-based benchmark for measuring how well image captions preserve image-level information for real downstream tasks.

---

## 🔥 News

- **[11/27/2025]** 🎉 [ArXiv paper](https://arxiv.org/abs/2511.21025) released!
- **[11/27/2025]** 📝 Blog post [English]() and [Chinese](https://zhuanlan.zhihu.com/p/1975613905834357034) released!
- **[11/27/2025]** 📊 [Validation set](https://huggingface.co/datasets/Borise/CaptionQA) released on HuggingFace!
- **[11/27/2025]** 💻 Draft code released! (Cleaning in progress - not yet compatible with HuggingFace dataset, meanwhile, please star our repo)

---

## 📎 Resources

- 📄 **Paper**: [CaptionQA: Is Your Caption as Useful as the Image Itself?](https://arxiv.org/abs/2511.21025)
- 📝 **Blog (Chinese / 中文博客)**: [从产业视角重新审视多模态：Caption这个多模态任务远超你的想象](https://zhuanlan.zhihu.com/p/1975613905834357034)
- 🤗 **Dataset on HuggingFace**: [Borise/CaptionQA](https://huggingface.co/datasets/Borise/CaptionQA)
- 🏆 **Leaderboard**: _coming soon_

---

## 🚀 Installation

### Prerequisites
- Python 3.8+
- API keys for the models you want to use (OpenAI, Google Gemini, or Anthropic Claude)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/CaptionQA.git
cd CaptionQA
```

2. **Install dependencies**
```bash
pip install openai google-genai anthropic pillow tqdm transformers
```

For vLLM support (optional):
```bash
pip install vllm
```

3. **Set up API keys**

Set your API keys as environment variables:
```bash
export OPENAI_API_KEY="your-openai-api-key"
export GOOGLE_API_KEY="your-google-api-key"  # For Gemini
export ANTHROPIC_API_KEY="your-anthropic-api-key"  # For Claude
```

4. **Verify installation**
```bash
python caption.py --list-prompts
```

> **Note**: The current code is under active development and is not yet fully compatible with the HuggingFace dataset format. Stay tuned for updates!

---
5. **Leaderboard Submission (Full Benchmark Evaluation)**

### Step 1: Run your model on the public validation subset

This ensures:

- correct caption format,

- correct mapping between image_id and caption,

- predictable caption length / style.

### Step 2: Prepare your caption file
Format can be JSONL or CSV, with the following fields:

| Field      | Description                           |
| ---------- | ------------------------------------- |
| `image_id` | The image identifier from the dataset |
| `caption`  | Your model-generated caption          |


### Step 3: Submit your captions
Send your caption file to:

captionqa.team@gmail.com

We will run:

- full taxonomy-aligned caption evaluation

- all domain subsets

- cross-domain utility metrics

- final leaderboard aggregation

Results are typically returned within 3–5 days, depending on queue time.

Your submission will be added to the official leaderboard once verified.

---

## 📚 Citation

If you use CaptionQA in your work, please cite:

```bibtex
@misc{yang2025captionqacaptionusefulimage,
      title={CaptionQA: Is Your Caption as Useful as the Image Itself?}, 
      author={Shijia Yang and Yunong Liu and Bohan Zhai and Ximeng Sun and Zicheng Liu and Emad Barsoum and Manling Li and Chenfeng Xu},
      year={2025},
      eprint={2511.21025},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.21025}, 
}

