# CaptionQA: Is Your Caption as Useful as the Image Itself?

<p align="center">
  <a href="https://captionqa.github.io/website"><b>🏆 Leaderboard</b></a> •
  <a href="https://huggingface.co/datasets/Borise/CaptionQA"><b>🤗 Dataset</b></a> •
  <a href="https://arxiv.org/abs/2511.21025"><b>📄 Paper</b></a>
</p>

> A utility-based benchmark for measuring how well image captions preserve image-level information for real downstream tasks.



## 🔥 News
- **[1/19/2026]** 🎉 CaptionQA is now integrated into [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/tasks/captionqa)! Evaluate your VLMs with a single command using the official benchmark framework.
- **[12/10/2025]** 🚀 Evaluation code cleaned and ready! Now fully compatible with HuggingFace dataset. Try it out and submit to our leaderboard!
- **[11/27/2025]** 🎉 [ArXiv paper](https://arxiv.org/abs/2511.21025) released!
- **[11/27/2025]** 📝 Blog post [English](https://huggingface.co/blog/Borise/rethinking-mm-from-industry-view) and [Chinese](https://zhuanlan.zhihu.com/p/1975613905834357034) released!
- **[11/27/2025]** 📊 [Validation set](https://huggingface.co/datasets/Borise/CaptionQA) released on HuggingFace!
- **[11/27/2025]** 💻 Draft code released! (Cleaning in progress - not yet compatible with HuggingFace dataset, meanwhile, please star our repo)


## 📎 Resources

- 📄 **Paper**: [CaptionQA: Is Your Caption as Useful as the Image Itself?](https://arxiv.org/abs/2511.21025)
- 📝 **Blog (English)**: [Rethinking Multimodality from an Industry Perspective](https://huggingface.co/blog/Borise/rethinking-mm-from-industry-view)
- 📝 **Blog (Chinese / 中文博客)**: [从产业视角重新审视多模态：Caption这个多模态任务远超你的想象](https://zhuanlan.zhihu.com/p/1975613905834357034)
- 🤗 **Dataset on HuggingFace**: [Borise/CaptionQA](https://huggingface.co/datasets/Borise/CaptionQA)
- 🏆 **Leaderboard**: [captionqa.github.io/website](https://captionqa.github.io/website)


## 🚀 Installation

### Prerequisites

Code is tested on **AMD MI325X** with the following Docker image:
```
rocm/vllm:rocm6.4.1_vllm_0.10.1_20250909
```
- ROCm 6.4.1
- vLLM 0.10.1

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/bronyayang/CaptionQA.git
cd CaptionQA
```

2. **Install dependencies**
```bash
pip install datasets pillow tqdm transformers
```

3. **Verify installation**
```bash
python qa.py --caption-path example_captions/simple/Qwen_Qwen2.5-VL-3B-Instruct.json --output-path ./results/Qwen2.5-VL-3B-Instruct.json
```

## 📊 Evaluation on CaptionQA Validation Set

### Step 1: Generate Captions

Generate captions for the validation set images. You can use:
- Our provided `caption.py` script
- Your own captioning code
- Your own prompts

As long as the output format matches this JSON structure:

```json
{
  "nat_001": "A gray cat sitting on fallen leaves...",
  "nat_002": "A forest path with autumn colors...",
  "doc_001": "A document showing a table with...",
  ...
}
```

Where keys are the `id` field from the dataset (e.g., `nat_001`, `doc_042`).

**Using our caption.py script:**

```bash
python caption.py \
    --output-dir ./example_captions \
    --split all \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --prompt SIMPLE
```

| Argument | Description |
|----------|-------------|
| `--output-dir` | Directory to save caption outputs |
| `--split` | Domain split: `natural`, `document`, `ecommerce`, `embodiedai`, or `all` |
| `--model` | Vision-language model for captioning |
| `--prompt` | Caption prompt style (use `--list-prompts` to see options) |

Output will be saved to: `./example_captions/<prompt_name>/<model_name>.json`

### Step 2: Evaluate Captions

Run QA evaluation using your generated captions:

```bash
python qa.py \
    --caption-path ./example_captions/simple/Qwen_Qwen2.5-VL-3B-Instruct.json \
    --output-path ./results/Qwen2.5-VL-3B-Instruct.json \
    --split all
```

**Arguments:**
| Argument | Description |
|----------|-------------|
| `--caption-path` | Path to your caption JSON file |
| `--output-path` | Path to save evaluation results |
| `--split` | Domain split to evaluate (must match caption split) |

### Step 3: View Results

The evaluation outputs accuracy and score metrics:

```
============================================================
Evaluation Results:
============================================================
Model: Qwen/Qwen2.5-72B-Instruct
Total questions: 8492
Correct answers: 4865 (57.29%)
'Cannot answer' selections: 2566
Total score: 5655.50 / 8492
Average score: 0.6660
============================================================
```

**Scoring Rules:**
- ✅ Correct answer: **1.0 point**
- ❌ Incorrect answer: **0.0 points**
- 🤷 "Cannot answer from caption": **1/n_choices + 0.05 points**

## ✨ Leaderboard Submission (Full Benchmark Evaluation)

### Step 1: Test on validation set first

Run your model on the public validation set (see above) to ensure correct caption format and mapping.

### Step 2: Submit your caption file

Send your caption JSON file (same format as validation set) to: **captionqa.team@gmail.com**

```json
{
  "nat_001": "Your caption for image nat_001...",
  "doc_001": "Your caption for image doc_001...",
  ...
}
```

We will run full evaluation across all domains and return results within 3–5 days.

### Step 3: Add your results to the leaderboard

Once you receive your evaluation results and are satisfied with them, add your model to the public leaderboard:

1. Fork the [leaderboard website repository](https://github.com/CaptionQA/website)

2. Edit `index.html` and add a new row to the leaderboard table in the `<tbody>` section (around line 208)

3. Copy this template and fill in your information:

```html
<tr>
  <td class="align-middle text-center">-<br><span class="eval-date">2025-Dec-02</span></td>
  <td class="align-middle text-center">Your Model Name<br><span class="affiliation">Your Organization</span></td>
  <td class="align-middle text-center"><span class="badge badge-primary">Proprietary</span></td>
  <!-- For open-source models, use: <span class="badge badge-success">Open-Source</span> -->
  <td class="align-middle text-center">7B</td>
  <td class="align-middle text-center">85.50</td>
  <td class="align-middle text-center">84.20</td>
  <td class="align-middle text-center">86.10</td>
  <td class="align-middle text-center">88.30</td>
  <td class="align-middle text-center">83.40</td>
</tr>
```

**How to fill in each field:**

| Field | What to put | Example |
|-------|-------------|---------|
| **Rank** | Use `-` (table auto-sorts by Overall score) | `-` |
| **Date** | Evaluation date (YYYY-MMM-DD) | `2025-Dec-02` |
| **Model Name** | Your model's name | `GPT-5` |
| **Organization** | Your affiliation (optional, use `-` if not provided) | `OpenAI` or `-` |
| **Type** | Proprietary or Open-Source (optional, use `-` if not provided) | `badge-primary`, `badge-success`, or `-` |
| **Size** | Model size (optional, use `-` if not provided) | `7B`, `72B`, or `-` |
| **Overall** | Overall score from our email | `85.50` |
| **Natural** | Natural domain score | `84.20` |
| **Document** | Document domain score | `86.10` |
| **E-comm** | E-commerce domain score | `88.30` |
| **Embodied** | Embodied AI domain score | `83.40` |

**Notes**:
- The main leaderboard displays Overall and 4 domain scores (Natural, Document, E-commerce, Embodied AI)
- Category-level scores are optional but recommended - they will be visible in the "Per Domain" tabs when users click on each domain
- Existing leaderboard entries keep their numeric ranks (1, 2, 3...). New submissions use `-` and the table auto-sorts by Overall score

<details>
<summary><b>Optional but recommended: Adding Category-Level Scores</b> (click to expand)</summary>

If you want to add category-level scores to the "Per Domain" tabs, you'll need to add rows to each domain table. To add category scores, find the corresponding domain table in `index.html` and add a row to each domain table you want to include.

<details>
<summary><b>Natural Domain Template</b></summary>

Search for `id="natural-board"` in `index.html` and add this row:

```html
<tr>
  <td class="align-middle text-center">-<br><span class="eval-date">2025-Dec-02</span></td>
  <td class="align-middle text-center">Your Model<br><span class="affiliation">Your Org</span></td>
  <td class="align-middle text-center"><span class="badge badge-primary">Proprietary</span></td>
  <td class="align-middle text-center">7B</td>
  <td class="align-middle text-center">85.50</td> <!-- Overall -->
  <td class="align-middle text-center">84.20</td> <!-- Action & Interaction -->
  <td class="align-middle text-center">83.50</td> <!-- Attribute -->
  <td class="align-middle text-center">86.30</td> <!-- Hallucination -->
  <td class="align-middle text-center">85.10</td> <!-- Object Existence -->
  <td class="align-middle text-center">84.70</td> <!-- Scene-Level -->
  <td class="align-middle text-center">82.90</td> <!-- Spatial -->
</tr>
```

</details>

<details>
<summary><b>Document Domain Template</b></summary>

Search for `id="document-board"` in `index.html` and add this row:

```html
<tr>
  <td class="align-middle text-center">-<br><span class="eval-date">2025-Dec-02</span></td>
  <td class="align-middle text-center">Your Model<br><span class="affiliation">Your Org</span></td>
  <td class="align-middle text-center"><span class="badge badge-primary">Proprietary</span></td>
  <td class="align-middle text-center">7B</td>
  <td class="align-middle text-center">86.10</td> <!-- Overall -->
  <td class="align-middle text-center">85.20</td> <!-- Chart-Specific -->
  <td class="align-middle text-center">87.30</td> <!-- Content-Level -->
  <td class="align-middle text-center">84.50</td> <!-- Diagram-Specific -->
  <td class="align-middle text-center">86.80</td> <!-- Domain-Specific -->
  <td class="align-middle text-center">85.90</td> <!-- Structural -->
  <td class="align-middle text-center">86.40</td> <!-- Table-Specific -->
</tr>
```

</details>

<details>
<summary><b>E-commerce Domain Template</b></summary>

Search for `id="ecommerce-board"` in `index.html` and add this row:

```html
<tr>
  <td class="align-middle text-center">-<br><span class="eval-date">2025-Dec-02</span></td>
  <td class="align-middle text-center">Your Model<br><span class="affiliation">Your Org</span></td>
  <td class="align-middle text-center"><span class="badge badge-primary">Proprietary</span></td>
  <td class="align-middle text-center">7B</td>
  <td class="align-middle text-center">88.30</td> <!-- Overall -->
  <td class="align-middle text-center">87.40</td> <!-- Brand & Marketing -->
  <td class="align-middle text-center">89.20</td> <!-- Contextual & Scene -->
  <td class="align-middle text-center">88.60</td> <!-- Functional -->
  <td class="align-middle text-center">87.80</td> <!-- Packaging -->
  <td class="align-middle text-center">89.10</td> <!-- Product-Level -->
  <td class="align-middle text-center">88.90</td> <!-- Textual Elements -->
  <td class="align-middle text-center">88.50</td> <!-- Visual Appearance -->
</tr>
```

</details>

<details>
<summary><b>Embodied AI Domain Template</b></summary>

Search for `id="embodiedai-board"` in `index.html` and add this row:

```html
<tr>
  <td class="align-middle text-center">-<br><span class="eval-date">2025-Dec-02</span></td>
  <td class="align-middle text-center">Your Model<br><span class="affiliation">Your Org</span></td>
  <td class="align-middle text-center"><span class="badge badge-primary">Proprietary</span></td>
  <td class="align-middle text-center">7B</td>
  <td class="align-middle text-center">83.40</td> <!-- Overall -->
  <td class="align-middle text-center">82.50</td> <!-- Activity & Task -->
  <td class="align-middle text-center">84.30</td> <!-- Functional & Semantic -->
  <td class="align-middle text-center">83.80</td> <!-- Perception -->
  <td class="align-middle text-center">82.90</td> <!-- Scene Dynamics -->
  <td class="align-middle text-center">83.70</td> <!-- Sensor & Embodiment -->
  <td class="align-middle text-center">84.10</td> <!-- Spatial & Environment -->
</tr>
```

</details>

</details>

4. Add your row anywhere in the `<tbody>` section (the table will auto-sort)

5. Submit a Pull Request to the leaderboard repository

We will review and merge your PR, and your results will appear on the [public leaderboard](https://captionqa.github.io/website).

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

