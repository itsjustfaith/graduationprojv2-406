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

  **Model file:** The YOLO model binary is intentionally NOT included in the repo (large file). Download `yolov8n.pt` and place it in `denco/` or set the `MODEL_PATH` environment variable to the model location. Example download sources: the Ultralytics releases or your preferred model provider.

  **Database & migrations:** The app uses SQLite (`db.sqlite`) and will auto-create tables on first run. Remove any committed `instance/db.sqlite` before publishing. Use Flask-Migrate to manage schema changes:

```bash
cd denco
flask db init   # only if migrations not initialized
flask db migrate -m "Init"
flask db upgrade
```

By default a local admin user will be created if no users exist. Set `DEFAULT_ADMIN_PASSWORD` in your environment (or `.env`) to customize the initial password.

- If YOLO initialization fails, confirm `yolov8n.pt` exists and that `ultralytics` and `torch` are compatible with your Python and CUDA versions.
- If YOLO initialization fails, confirm `yolov8n.pt` exists at the path set by `MODEL_PATH` or in `denco/`, and that `ultralytics` and `torch` are compatible with your Python and CUDA versions.
- Security: Set `SECRET_KEY` in your environment (see `.env.example`). Do not commit secrets into the repo.
- For camera/live feed: the app tries local camera indices `1` then `0` by default. For IP cameras provide an RTSP URL (format shown in the web UI) when selecting `ip_camera`.
- If using a GPU, install a matching `torch` build (see https://pytorch.org) before installing the other requirements.
