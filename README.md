# AI Drone Navigation (sp 1)

Setup (Windows):
    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt

Run:
    python navigation.py                 # logic tests, no camera needed
    python main.py                       # OpenCV window demo (--model rtdetr-l --mode baseline)
    python app.py                        # website -> http://localhost:5000  (click "Start camera")
    python benchmark.py video.mp4        # all models x modes -> benchmark_results.csv
