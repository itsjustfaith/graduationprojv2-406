##DENCO - People Detection and Counting System (2025)

Steps on how to run the system:

## Requirements

- Python 3.8 - 3.11
- Git
- (optional) CUDA GPU for faster YOLO inference
- `denco/requirements.txt` contains Python dependencies

## Setup and Run

1. Open a terminal and go to the repo root:
   ```bash
   cd denco
   ```
2. Create and activate a virtual environment:
   - PowerShell:
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
   - CMD:
     ```cmd
     python -m venv .venv
     .\.venv\Scripts\activate.bat
     ```
3. Install dependencies:
   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. Copy the example env file and set values:
   ```bash
   copy ..\.env.example .env
   ```
   Then edit `.env` and set `SECRET_KEY`, `DEFAULT_ADMIN_PASSWORD`, and `MODEL_PATH` if needed.
5. Download `yolov8n.pt` and place it in `denco/`, or set `MODEL_PATH` to the model location.
6. Initialize the database if needed, then upgrade migrations:
   ```bash
   python -m flask --app app2 db upgrade
   ```
7. Start the app:
   ```bash
   python app2.py
   ```
8. Open your browser at:
   ```text
   http://localhost:5000
   ```

## Notes

- The YOLO model file should not be stored in the repo. Download it separately.
- The app creates a local SQLite database automatically.
- Set `SECRET_KEY` in `.env` for secure sessions.
- If no users exist, the app creates a default admin account using `DEFAULT_ADMIN_PASSWORD`.
- If you use an RTSP camera, enter the correct camera URL in the web UI.
- If using a GPU, install a matching `torch` build from https://pytorch.org.
