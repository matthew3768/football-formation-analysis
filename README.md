# Football Tactical Analysis

Automated football tactical analysis from broadcast match footage. The project tracks players, assigns teams from kit colours, estimates formations, and presents the results in a Streamlit dashboard.

## Main Features

- Upload a football video clip through a web dashboard.
- Track players using YOLOv8 and ByteTrack.
- Clean tracking data and select a stable analysis window.
- Cluster players into teams using kit colour information.
- Detect observed team formations.
- Summarise attacking and defending shapes from phase windows.
- Export team clustering and phase-analysis data.

## Project Structure

```text
football-formation-analysis/
|-- app/
|   `-- app.py                  # Streamlit dashboard
|-- data/
|   `-- .gitkeep                # Runtime uploads are written here
|-- outputs/
|   `-- .gitkeep                # Runtime analysis outputs are written here
|-- scripts/
|   `-- download_data.py
|-- src/
|   |-- detection/
|   |-- formation/
|   |-- team_assignment/
|   |-- tracking/
|   `-- utils/
|-- main.py                     # Script pipeline for local development
|-- requirements.txt
`-- yolov8s.pt                  # YOLO model weights used by the dashboard
```

## Requirements

- Python 3.10 or later
- FFmpeg installed and available on the system `PATH`
- The Python packages listed in `requirements.txt`

FFmpeg is required because the dashboard converts generated OpenCV videos into browser-compatible MP4 files.

## Setup

From the project folder:

```bash
python -m venv venv
```

Activate the virtual environment.

On Windows:

```bash
venv\Scripts\activate
```

On macOS/Linux:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Dashboard

Start the Streamlit app from the project folder:

```bash
streamlit run app/app.py
```

Then open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

Upload a short football clip using the dashboard. The app will generate tracking video, formation results, team clustering outputs, and downloadable summary files.

## Model Weights

The dashboard uses `yolov8s.pt` from the project root. Include this file in the submitted zip if possible so the project can run without needing to download model weights on first use.

If `yolov8s.pt` is missing, Ultralytics may attempt to download it automatically, which requires an internet connection.

## Notes on `main.py`

The recommended way to run the project is the Streamlit dashboard:

```bash
streamlit run app/app.py
```

`main.py` is kept as a development script and expects a local clip at:

```text
data/clips/best_segment.mp4
```

That clip is not required for the dashboard because users can upload a clip directly.

## Outputs

After analysis, outputs are written under:

```text
outputs/streamlit/
```

The dashboard also provides download buttons for:

- team cluster CSV
- phase summary JSON
- phase window CSV

## Limitations

- Results depend on the quality, camera angle, and length of the uploaded clip.
- Clips with too few visible players may not produce reliable formation or phase summaries.
- Processing can take several minutes depending on hardware and clip length.
