# Dashboard refresh pipeline

Regenerates and republishes the AI Market Regime Forecaster artifact.

Requires a Guardian Open Platform API key (free, https://open-platform.theguardian.com/access/) in the `GUARDIAN_API_KEY` environment variable before running step 4 below — e.g. `export GUARDIAN_API_KEY=your-key-here` (or set it in your shell profile / `.env` file, which is gitignored). Without it, step 4 logs a warning and writes an empty `news_candidates.json` rather than failing.

Run in order:

1. `python ../scripts/fetch_data.py` — re-fetch daily OHLCV, 2007-present
2. `python ../scripts/train_autoencoder.py` — retrain (deterministic seed) on the frozen 2007-2021 train window, apply forward
3. `python ../scripts/fit_hmm.py` — refit the HMM on the same frozen train window, label the full history
4. `python build_data.py` — mechanical extraction: writes `regime_data.json` (everything about the model's own state) and `news_candidates.json` (~40 unscored recent market/macro Guardian headlines)
5. **Judgment step, done by whichever Claude session runs this**, not a script:
   - Read `news_candidates.json`, pick the 5 headlines most relevant to the *current* regime call (not just the newest 5)
   - Write a 2-3 sentence forecast-take paragraph grounded in `regime_data.json`'s sector readings + the picked headlines
   - Assemble `news_data.json` in the shape `{"cache_as_of": "...", "recent": [{date,title,url,section}, ...5 items], "transition_news": [...carry forward from the previous news_data.json if present, else []]}`
   - Fill `template.html`'s three placeholders (`__REGIME_DATA__`, `__NEWS_DATA__`, `__LLM_TAKE__` — the last one is a JSON-encoded string) with `regime_data.json`, the assembled `news_data.json`, and the take paragraph, and save as `output.html`
   - Publish `output.html` via the Artifact tool with `url` set to the existing published artifact URL, so it updates in place

Steps 1-4 are pure mechanics and safe to always run exactly the same way. Step 5 is where the actual reasoning happens each cycle — deliberately left to the agent rather than a scoring formula (see the project's NLP-robustness review for why a keyword-score reused from a different project would misrepresent relevance here).
