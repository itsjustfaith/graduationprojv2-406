# graduationprojv2-406

Steps on how to run the system:

<img width="620" height="698" alt="image" src="https://github.com/user-attachments/assets/59925db3-ae24-4efb-a7c7-0fce5dc9780f" />
<img width="623" height="713" alt="image" src="https://github.com/user-attachments/assets/3f102ec6-e9ad-4c0a-b485-015afe7441f6" />

**Run Instructions**

- **Prerequisites:** Python 3.8 - 3.11, Git, (optional) CUDA-enabled GPU for faster inference.
  -- **Dependencies:** See `denco/requirements.txt` for required Python packages.

- **Quick start (Windows PowerShell)**:

```powershell
cd denco
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
python app2.py
```

- **Quick start (Windows CMD)**:

```cmd
cd denco
python -m venv .venv
.\.venv\Scripts\activate.bat
pip install --upgrade pip
pip install -r requirements.txt
python app2.py
```

- The app starts on http://0.0.0.0:5000 — open http://localhost:5000 in your browser.

- **Model file:** `yolov8n.pt` should be placed in `denco/` (it is included in the repo). If you want a different YOLOv8 model, replace that filename and ensure `app2.py` references it.

- **Database:** The app uses SQLite (`db.sqlite`) and will auto-create tables on first run. A default admin user is created automatically with email `admin@example.com` and password `admin123` if no users exist.

- **Notes / Troubleshooting:**
  - If YOLO initialization fails, confirm `yolov8n.pt` exists and that `ultralytics` and `torch` are compatible with your Python and CUDA versions.
  - For camera/live feed: the app tries local camera indices `1` then `0` by default. For IP cameras provide an RTSP URL (format shown in the web UI) when selecting `ip_camera`.
  - If using a GPU, install a matching `torch` build (see https://pytorch.org) before installing the other requirements.
