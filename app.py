import os
import pickle
import numpy as np
from flask import Flask, render_template_string, request

app = Flask(__name__)

# Build dynamic absolute path to ensure file loading works on Render
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Check for 'SVM_model_pkl' or fallback to 'SVM_model.pkl'
MODEL_PATH = os.path.join(BASE_DIR, "SVM_model_pkl")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(BASE_DIR, "SVM_model.pkl")

# Load trained SVM model
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Front-End UI Template with CSS & Animations Embedded
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SVM Prediction Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-gradient: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #311042 100%);
            --card-bg: rgba(255, 255, 255, 0.05);
            --card-border: rgba(255, 255, 255, 0.12);
            --accent-color: #6366f1;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', sans-serif;
        }

        body {
            background: var(--bg-gradient);
            color: var(--text-primary);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 2rem 1rem;
        }

        .container {
            width: 100%;
            max-width: 850px;
            background: var(--card-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--card-border);
            border-radius: 24px;
            padding: 2.5rem;
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.4);
            animation: fadeIn 0.8s ease-out forwards;
        }

        header {
            text-align: center;
            margin-bottom: 2rem;
        }

        header h1 {
            font-size: 2.2rem;
            font-weight: 700;
            background: linear-gradient(90deg, #818cf8, #c084fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }

        header p {
            color: var(--text-secondary);
            font-size: 0.95rem;
        }

        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
            gap: 1.25rem;
        }

        .input-group {
            display: flex;
            flex-direction: column;
            animation: slideUp 0.6s ease-out ease-in-out;
        }

        .input-group label {
            font-size: 0.85rem;
            font-weight: 600;
            margin-bottom: 0.4rem;
            color: #cbd5e1;
        }

        .input-group input, .input-group select {
            width: 100%;
            padding: 0.75rem 1rem;
            border-radius: 12px;
            border: 1px solid var(--card-border);
            background: rgba(15, 23, 42, 0.6);
            color: #fff;
            font-size: 0.95rem;
            outline: none;
            transition: all 0.3s ease;
        }

        .input-group input:focus, .input-group select:focus {
            border-color: var(--accent-color);
            box-shadow: 0 0 12px rgba(99, 102, 241, 0.4);
            background: rgba(15, 23, 42, 0.8);
        }

        .btn-submit {
            grid-column: 1 / -1;
            margin-top: 1rem;
            padding: 0.9rem;
            border-radius: 12px;
            border: none;
            background: linear-gradient(90deg, #6366f1, #a855f7);
            color: white;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }

        .btn-submit:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(99, 102, 241, 0.3);
        }

        .btn-submit:active {
            transform: translateY(0);
        }

        .result-box {
            margin-top: 2rem;
            padding: 1.5rem;
            border-radius: 16px;
            background: rgba(99, 102, 241, 0.15);
            border: 1px solid rgba(99, 102, 241, 0.3);
            text-align: center;
            animation: popIn 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        }

        .result-box h3 {
            font-size: 1.1rem;
            color: #c7d2fe;
            margin-bottom: 0.3rem;
        }

        .result-box .prediction {
            font-size: 1.8rem;
            font-weight: 700;
            color: #38bdf8;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @keyframes slideUp {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @keyframes popIn {
            from { opacity: 0; transform: scale(0.9); }
            to { opacity: 1; transform: scale(1); }
        }
    </style>
</head>
<body>

    <div class="container">
        <header>
            <h1>SVM Model Inference</h1>
            <p>Input parameters to run predictions on your trained model</p>
        </header>

        <form method="POST" action="/predict">
            <div class="form-grid">
                <div class="input-group">
                    <label for="gender">Gender</label>
                    <select id="gender" name="gender" required>
                        <option value="0">Female</option>
                        <option value="1">Male</option>
                    </select>
                </div>

                <div class="input-group">
                    <label for="age">Age</label>
                    <input type="number" id="age" name="age" placeholder="e.g., 20" step="any" required>
                </div>

                <div class="input-group">
                    <label for="study_hours_per_week">Study Hours / Week</label>
                    <input type="number" id="study_hours_per_week" name="study_hours_per_week" placeholder="e.g., 15" step="any" required>
                </div>

                <div class="input-group">
                    <label for="attendance_rate">Attendance Rate (%)</label>
                    <input type="number" id="attendance_rate" name="attendance_rate" placeholder="e.g., 85.5" step="any" required>
                </div>

                <div class="input-group">
                    <label for="parent_education">Parent Education Level</label>
                    <select id="parent_education" name="parent_education" required>
                        <option value="0">High School</option>
                        <option value="1">Bachelor's</option>
                        <option value="2">Master's / Higher</option>
                    </select>
                </div>

                <div class="input-group">
                    <label for="internet_access">Internet Access</label>
                    <select id="internet_access" name="internet_access" required>
                        <option value="1">Yes</option>
                        <option value="0">No</option>
                    </select>
                </div>

                <div class="input-group">
                    <label for="extracurricular">Extracurricular Activities</label>
                    <select id="extracurricular" name="extracurricular" required>
                        <option value="1">Yes</option>
                        <option value="0">No</option>
                    </select>
                </div>

                <div class="input-group">
                    <label for="previous_score">Previous Score</label>
                    <input type="number" id="previous_score" name="previous_score" placeholder="e.g., 78.0" step="any" required>
                </div>

                <div class="input-group">
                    <label for="final_score">Final Score</label>
                    <input type="number" id="final_score" name="final_score" placeholder="e.g., 82.5" step="any" required>
                </div>

                <button type="submit" class="btn-submit">Generate Prediction ✨</button>
            </div>
        </form>

        {% if prediction %}
        <div class="result-box">
            <h3>Prediction Result</h3>
            <div class="prediction">{{ prediction }}</div>
        </div>
        {% endif %}
    </div>

</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = [
            float(request.form["gender"]),
            float(request.form["age"]),
            float(request.form["study_hours_per_week"]),
            float(request.form["attendance_rate"]),
            float(request.form["parent_education"]),
            float(request.form["internet_access"]),
            float(request.form["extracurricular"]),
            float(request.form["previous_score"]),
            float(request.form["final_score"])
        ]
        
        input_data = np.array([features])
        prediction = model.predict(input_data)[0]
        
        return render_template_string(HTML_TEMPLATE, prediction=str(prediction))
    except Exception as e:
        return render_template_string(HTML_TEMPLATE, prediction=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
