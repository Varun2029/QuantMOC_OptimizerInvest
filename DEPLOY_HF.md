# Deploy to Hugging Face Spaces (Free)

## 1. Push your code to GitHub

Create a repo and push:

```bash
cd c:\Users\varun\OneDrive\Desktop\Projects\QUANT
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/QUANT.git
git push -u origin main
```

## 2. Create a Hugging Face Space

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. Click **Create new Space**
3. Choose:
   - **Name:** quant-hedge-fund
   - **License:** MIT
   - **SDK:** **Streamlit**
   - **Visibility:** Public
4. Click **Create Space**

## 3. Connect GitHub

1. In your Space, click **Settings** (gear icon)
2. Under "Repository", click **Connect to Hub**
3. Or: **Settings → Clone your Space** and push your code
4. Easiest: **Create new Space** → select **Import from a Git repository**
   - Paste: `https://github.com/YOUR_USERNAME/QUANT`
   - HF will use `app.py` and `requirements.txt` from the repo

## 4. Or: Create Space and upload files manually

1. Create a **new Space** with SDK **Streamlit**
2. Upload your files (or use the built-in editor):
   - `app.py` (must be at root)
   - `requirements.txt`
   - All folders: `config.py`, `data/`, `feature_engine/`, `regime_model/`, `optimizer/`, `risk_engine/`, `backtester/`, `execution_simulator/`

## 5. Build and run

- HF builds and runs: `streamlit run app.py`
- First load may take 1–2 minutes (installs deps + fetches data)
- Click **Run Full Pipeline** to run the backtest

## Notes

- **Free tier:** 2 CPU, 16GB RAM, public Spaces
- **Data:** Fetches live via yfinance on each run
- If build fails, check that `requirements.txt` has pinned versions
