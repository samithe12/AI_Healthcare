# ❤️ AI Heart Disease Risk Predictor

## 🚀 One-Click Setup

### 1. Clone & Setup (Run These Commands)
```bash
git clone https://github.com/yourusername/ai-healthcare-project.git
cd ai-healthcare-project

# Windows (PowerShell):
python -m venv .venv
.\.venv\Scripts\Activate
pip install -r requirements.txt || (pip install numpy==1.26.4 scipy==1.11.4 --prefer-binary && pip install -r requirements.txt)

# Mac/Linux:
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
# ➡️ Open http://localhost:8501
# If Python version issues:
brew install python@3.10
python3.10 -m venv .venv
