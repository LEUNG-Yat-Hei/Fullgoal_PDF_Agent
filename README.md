# image-agent3

Extracts decision-relevant information from a research PDF into an HTML report with charts.

Put `DEEPSEEK_API_KEY=...` in `.env` (or set it as an environment variable when deployed).

## Web app

```
uv run streamlit run app.py
```

Open http://localhost:8501, upload a PDF and click **Generate report**. Others on the same network can use
`http://<your-ip>:8501`.

## Command line

```
uv run main.py test2.pdf
```

Saves the report as `test2.html` in the project folder.
