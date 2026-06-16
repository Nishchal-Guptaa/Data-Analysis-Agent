## 📄 Document Analysis Agent 🤖

A powerful AI-powered web application that analyzes structured and unstructured documents (CSV, Excel, Word, PDF, text, images with OCR) and provides intelligent summaries, insights, and visualizations using the **Together.ai LLaMA-4 Maverick model**.

---

### 🚀 Features

* 📁 Upload multiple document formats: `.csv`, `.xlsx`, `.pdf`, `.txt`, `.docx`, `.png`, `.jpg`, etc.
* 🔍 Automatically extracts and analyzes content from documents.
* 🧠 Uses LLaMA-4 to answer questions about the uploaded data or content.
* 📊 Generates interactive charts (histogram, scatter, bar, line, heatmap, boxplot).
* 📷 OCR-powered image-to-text extraction and structured table recognition.
* 🗨️ Stores and displays conversational history with the LLM agent.

---

### 🛠️ Tech Stack

* **Frontend**: [Streamlit](https://streamlit.io/)
* **Backend / LLM**: [Huggingface](https://huggingface.co/docs/inference-providers/en/index)
* **OCR**: `pytesseract`
* **Document Processing**: `pdfplumber`, `docx2txt`, `pandas`, `openpyxl`
* **Visualization**: `Plotly`, `Seaborn`, `Matplotlib`

---

### 📦 Installation

1. **Clone the repository:**

```bash
git clone https://github.com/Nishchal-Guptaa/Data-Analysis-Agent.git
cd Data-Analysis-Agent
```

2. **Create a virtual environment (optional but recommended):**

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

4. **Set up your HF Token:**

Create a `.streamlit/secrets.toml` file (not tracked by Git):

```toml
HF_API_KEY = "your_hf_token"
```

---

### ▶️ Run the App

```bash
streamlit run model.py
```

Then open `http://localhost:8501` in your browser.

---

### 🧪 Supported File Types

| File Type                     | Features                                           |
| ----------------------------- | -------------------------------------------------- |
| `.csv`, `.xlsx`               | Parses structured data and supports visualizations |
| `.txt`, `.docx`, `.pdf`       | Text extraction, summary, and LLM analysis         |
| `.png`, `.jpg`, `.tiff`, etc. | OCR-based text extraction + table detection        |
| OCR Output                    | Converted to structured tables if applicable       |

---

### 🧠 Example Queries

* "Summarize the dataset."
* "Find the strongest correlations."
* "Suggest relevant charts."
* "What type of document is this image?"
* "Extract key information from this PDF."

---

### 📂 Project Structure

```
├── model.py                # Main Streamlit app
├── requirements.txt        # Python dependencies
├── .gitignore              # Ignored files and folders
└── .streamlit/
    └── secrets.toml        # Your Together API key (DO NOT COMMIT)
```

---

### 🛡️ Security Note

> Never commit `.streamlit/secrets.toml` or any file containing your API keys. It is already ignored via `.gitignore`.

---

### 🙌 Credits

* Built by \Nishchal Gupta
* Powered by [huggingface](https://huggingface.co/docs/inference-providers/en/index)

---

### Checkout the Agent [here](https://data-analysis-agent-jwgkfarf964ru9vlkdfhst.streamlit.app/)
