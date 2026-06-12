# CCM-RWI Interactive Analysis

Streamlit web app for the CCM Risk-Weighted Impact framework. Users (clinicians or patients) can modify the input parameters and see personalized projected outcomes plus the decision-zone heatmap with crossover annotations.

## Local development

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Files

- `app.py` — Streamlit UI
- `engine.py` — Forward-Markov engine + decision-zone heatmap
- `reference_data/` — SSA actuarial life tables (sex-stratified)
- `requirements.txt`

## Deployment

The app is self-contained and can be deployed on any service that runs Streamlit. The simplest options:

- **Streamlit Community Cloud** (free): connect a GitHub repository, point at `app.py`, and it deploys automatically
- **Hugging Face Spaces**: upload as a Streamlit Space
- **Render / Railway / Fly.io**: standard Python web-app deployments

For all options, the working directory should contain `app.py`, `engine.py`, `requirements.txt`, and the `reference_data/` folder.
