# BugSpot Web Parameter Playground

This app gives you a local UI to:

1. Upload an example video
2. Tune BugSpot config values with sliders
3. Click **Update Config**
4. Click **Process Video**
5. Watch an annotated result video with detected boxes
6. Download your tuned config YAML

## Run

From the repository root:

```bash
cd website
pip install -r requirements.txt
streamlit run app.py
```

The app opens in your browser and runs locally.

## Notes

- The app imports BugSpot from `../bugspot-main/src`.
- Processing artifacts (annotated video, crops, composites) are written to temp folders and shown in the UI.
- Confirmed tracks are shown in green in the output video; candidate (unconfirmed) detections are shown in orange.
