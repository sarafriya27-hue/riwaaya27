# Riwaaya Analytics Dashboard

An interactive Streamlit dashboard analyzing a 1,000-response consumer survey for **Riwaaya**, a
proposed premium, authenticated Indian handicrafts marketplace. Covers descriptive analytics,
diagnostic analytics, supervised classification (membership interest prediction), regression-based
predictive analytics (spending and purchase frequency forecasting), and prescriptive business strategy.

## Why this is one file

This project is deliberately built as a **single Python file with no subfolders**, other than the
data file sitting right next to it. The earlier version split shared code into a separate package
folder, which ran perfectly locally but failed on Streamlit Cloud with `ModuleNotFoundError`, almost
always a sign that a subfolder didn't make it into the GitHub repo during upload (a common issue with
GitHub's drag-and-drop web uploader, which doesn't reliably preserve nested folder structures in every
browser). A single flat file removes that failure point entirely: there's nothing for an upload step to
misplace.

Navigation between sections is handled with a sidebar menu inside the one file, instead of Streamlit's
folder-based multipage system.

## Project structure

```
riwaaya_dashboard/
├── app.py                          # Everything: data loading, modeling, all six views
├── riwaaya_survey_cleaned.csv      # Cleaned survey data (1,000 rows, 56 columns)
├── requirements.txt
└── README.md
```

That's it. Only two files actually need to be uploaded for the app to run: `app.py` and
`riwaaya_survey_cleaned.csv`, plus `requirements.txt` so Streamlit Cloud knows what to install.

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`. The first visit to "Predictive Modeling" or "Predictive Analytics"
in the sidebar takes 60-100 seconds while the models grid-search and cross-validate; results are
cached afterward via `st.cache_resource`, so it only happens once per session.

## Deploying to Streamlit Community Cloud via GitHub

1. **Create a new GitHub repository.** The simplest way to avoid any upload issues: use `git` from a
   terminal rather than the GitHub website's drag-and-drop uploader.
   ```bash
   git init
   git add .
   git commit -m "Riwaaya analytics dashboard"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo>.git
   git push -u origin main
   ```
   If you don't have `git` set up, GitHub Desktop (a free app) is the next most reliable option, drag
   the whole project folder into it and commit/push from there, rather than using the browser upload
   button.
2. **Verify on github.com** that both `app.py` and `riwaaya_survey_cleaned.csv` actually appear in the
   repo, at the root, before deploying. This single check would have caught the previous error.
3. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub, click **New app**,
   select the repository and branch, and set the main file path to `app.py`.
4. Click **Deploy**.

## Methodology notes

- **Classification target:** `target_membership_interest`, 1 if a respondent said "Yes, definitely" to
  joining a membership program, 0 otherwise. Close to perfectly balanced (about 50.5% / 49.5%).
- **Feature set:** the ground-truth segment label (`segment_true`) is excluded from every model, since
  it would leak the answer.
- **Tuning:** all four classifiers (KNN, Decision Tree, Random Forest, Gradient Boosting) are tuned
  with `GridSearchCV` under 5-fold stratified cross-validation, optimizing ROC-AUC.
- **Regression targets:** annual spending potential and future purchase frequency each exclude their
  own direct algebraic components from the feature set, to avoid a circular, trivially "perfect" fit.
- **Trade-off of going single-file:** the Predictive Modeling and Predictive Analytics sections each
  train their own copy of the classification model rather than sharing one cached object across
  sections (each section's own cache still makes repeat visits to that section instant within a
  session). This costs a bit of redundant computation in exchange for a deployment structure with
  nothing left to go wrong.
