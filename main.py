from flask import Flask, render_template, request, redirect, session
import mysql.connector
from flask_bcrypt import Bcrypt
import pandas as pd

# New imports for the machine learning model
import pickle
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# --- MOCK ML MODEL SETUP ---
# In a real-world scenario, you would have a pre-trained model.
# We are creating a dummy one here to make the example runnable.
# This model predicts a value based on two input features.

def create_and_save_mock_model():
    """
    Trains a simple Linear Regression model on dummy data and saves it.
    This function is for demonstration purposes.
    """
    print("Creating and saving mock ML model...")
    # Dummy data
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
    y = np.array([5, 8, 11, 14, 17]) # y = 3*x1 - x2 + 4

    # Train the model
    model = LinearRegression()
    model.fit(X, y)

    # Save the model to a pickle file
    model_filename = 'model.pkl'
    with open(model_filename, 'wb') as file:
        pickle.dump(model, file)
    print(f"Mock model saved as '{model_filename}'")

# Call the function to create the mock model when the script starts
create_and_save_mock_model()

# --- FLASK APPLICATION SETUP ---

app = Flask(__name__)
app.secret_key = "loke2005@."
bcrypt = Bcrypt(app)

# MySQL connection
def get_db_connection():
    # You will need to configure these database credentials.
    # For this example, we'll assume a connection can be made.
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="loke2005@.",
        database="myappdb"
    )

# --- ROUTES ---

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATES['index.html'])

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        password = bcrypt.generate_password_hash(request.form["password"]).decode("utf-8")

        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("INSERT INTO users (username, email, password) VALUES (%s, %s, %s)",
                           (username, email, password))
            conn.commit()
        except Exception as e:
            # Add error handling in case of DB issues
            print(f"Database error during signup: {e}")
            return "Signup failed. Please try again."
        finally:
            cursor.close()
            conn.close()

        return redirect("/login")
    return render_template_string(HTML_TEMPLATES['signup.html'])

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        try:
            conn = get_db_connection()
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT * FROM users WHERE email=%s", (email,))
            user = cursor.fetchone()
        except Exception as e:
            print(f"Database error during login: {e}")
            user = None
        finally:
            cursor.close()
            conn.close()

        if user and bcrypt.check_password_hash(user["password"], password):
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            return redirect("/dashboard")
        else:
            return "Invalid credentials"
    return render_template_string(HTML_TEMPLATES['login.html'])

@app.route("/dashboard")
def dashboard():
    if "user_id" in session:
        return render_template_string(HTML_TEMPLATES['dashboard.html'], username=session["username"])
    return redirect("/login")

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")

# --- NEW ML PREDICTION ROUTE ---

# Load the pre-trained model once when the app starts
model = None
try:
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    print("ML model loaded successfully.")
except FileNotFoundError:
    print("Error: 'model.pkl' not found. Ensure the model file exists.")
    model = None
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        # Render upload page
        return render_template_string(HTML_TEMPLATES['predict.html'])

    try:
        if "file" not in request.files:
            return render_template_string(
                HTML_TEMPLATES['predict.html'],
                prediction_result="No file part in the request"
            )

        file = request.files["file"]

        if file.filename == "":
            return render_template_string(
                HTML_TEMPLATES['predict.html'],
                prediction_result="No file selected"
            )

        # Read file into pandas DataFrame
        df = pd.read_csv(file)  # adjust if Excel (use pd.read_excel)

        # Run prediction using ML model
        predictions = model.predict(df)

        # Save predictions into CSV (optional)
        output_df = pd.DataFrame(predictions, columns=["Prediction"])
        output_csv = "predictions.csv"
        output_df.to_csv(output_csv, index=False)

        return render_template_string(
            HTML_TEMPLATES['predict.html'],
            prediction_result=f"✅ Prediction complete! {len(predictions)} rows processed. Download CSV from dashboard."
        )

    except Exception as e:
        return render_template_string(
            HTML_TEMPLATES['predict.html'],
            prediction_result=f"❌ An error occurred during prediction: {str(e)}"
        )

# --- HTML TEMPLATES (using render_template_string for a single file) ---

from jinja2 import Environment, FileSystemLoader

# A simple way to serve templates from memory for this single-file example
def render_template_string(template_string, **kwargs):
    env = Environment(loader=FileSystemLoader('.'))
    return env.from_string(template_string).render(**kwargs)

HTML_TEMPLATES = {
    'index.html': """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Home</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap">
    <style>body { font-family: 'Inter', sans-serif; }</style>
</head>
<body class="bg-gray-100 min-h-screen flex items-center justify-center">
    <div class="bg-white p-8 rounded-xl shadow-lg w-full max-w-md text-center space-y-6">
        <h1 class="text-4xl font-bold text-gray-800">Welcome</h1>
        <p class="text-gray-600">Please choose an option to continue.</p>
        <div class="space-y-4">
            <a href="/login" class="block w-full bg-blue-600 text-white font-semibold py-3 px-4 rounded-lg shadow-md hover:bg-blue-700 transition-colors">
                Log In
            </a>
            <a href="/signup" class="block w-full bg-green-600 text-white font-semibold py-3 px-4 rounded-lg shadow-md hover:bg-green-700 transition-colors">
                Sign Up
            </a>
            <a href="/predict" class="block w-full bg-purple-600 text-white font-semibold py-3 px-4 rounded-lg shadow-md hover:bg-purple-700 transition-colors">
                Make a Prediction
            </a>
        </div>
    </div>
</body>
</html>
""",
    'signup.html': """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sign Up</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap">
    <style>body { font-family: 'Inter', sans-serif; }</style>
</head>
<body class="bg-gray-100 min-h-screen flex items-center justify-center">
    <div class="bg-white p-8 rounded-xl shadow-lg w-full max-w-md">
        <h1 class="text-3xl font-bold text-center text-gray-800 mb-6">Create an Account</h1>
        <form action="/signup" method="post" class="space-y-4">
            <div>
                <label for="username" class="block text-sm font-medium text-gray-700">Username</label>
                <input type="text" id="username" name="username" required
                       class="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500">
            </div>
            <div>
                <label for="email" class="block text-sm font-medium text-gray-700">Email</label>
                <input type="email" id="email" name="email" required
                       class="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500">
            </div>
            <div>
                <label for="password" class="block text-sm font-medium text-gray-700">Password</label>
                <input type="password" id="password" name="password" required
                       class="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500">
            </div>
            <button type="submit"
                    class="w-full bg-green-600 text-white font-semibold py-2 px-4 rounded-lg shadow-md hover:bg-green-700 transition-colors">
                Sign Up
            </button>
        </form>
        <p class="mt-4 text-center text-gray-600">
            Already have an account? <a href="/login" class="text-blue-600 hover:underline">Log In</a>
        </p>
    </div>
</body>
</html>
""",
    'login.html': """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Log In</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap">
    <style>body { font-family: 'Inter', sans-serif; }</style>
</head>
<body class="bg-gray-100 min-h-screen flex items-center justify-center">
    <div class="bg-white p-8 rounded-xl shadow-lg w-full max-w-md">
        <h1 class="text-3xl font-bold text-center text-gray-800 mb-6">Log In</h1>
        <form action="/login" method="post" class="space-y-4">
            <div>
                <label for="email" class="block text-sm font-medium text-gray-700">Email</label>
                <input type="email" id="email" name="email" required
                       class="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500">
            </div>
            <div>
                <label for="password" class="block text-sm font-medium text-gray-700">Password</label>
                <input type="password" id="password" name="password" required
                       class="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500">
            </div>
            <button type="submit"
                    class="w-full bg-blue-600 text-white font-semibold py-2 px-4 rounded-lg shadow-md hover:bg-blue-700 transition-colors">
                Log In
            </button>
        </form>
        <p class="mt-4 text-center text-gray-600">
            Don't have an account? <a href="/signup" class="text-green-600 hover:underline">Sign Up</a>
        </p>
    </div>
</body>
</html>
""",
    'dashboard.html': """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap">
    <style>body { font-family: 'Inter', sans-serif; }</style>
</head>
<body class="bg-gray-100 min-h-screen flex items-center justify-center">
    <div class="bg-white p-8 rounded-xl shadow-lg w-full max-w-lg text-center space-y-6">
        <h1 class="text-4xl font-bold text-gray-800">Hello, {{ username }}!</h1>
        <p class="text-gray-600">Welcome to your dashboard.</p>
        <div class="space-y-4">
            <a href="/predict" class="block w-full bg-purple-600 text-white font-semibold py-3 px-4 rounded-lg shadow-md hover:bg-purple-700 transition-colors">
                Go to Prediction Page
            </a>
            <a href="/logout" class="block w-full bg-red-600 text-white font-semibold py-3 px-4 rounded-lg shadow-md hover:bg-red-700 transition-colors">
                Log Out
            </a>
        </div>
    </div>
</body>
</html>
""",
    'predict.html': """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Upload File for Prediction</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap">
    <style>
        body { font-family: 'Inter', sans-serif; }
        .drag-area {
            border: 2px dashed #7e22ce;
            border-radius: 12px;
            background: #faf5ff;
            padding: 40px;
            text-align: center;
            transition: 0.3s;
        }
        .drag-area.dragover {
            background: #ede9fe;
            border-color: #5b21b6;
        }
    </style>
</head>
<body class="bg-gray-100 min-h-screen flex items-center justify-center">
    <div class="bg-white p-8 rounded-xl shadow-lg w-full max-w-md">
        <h1 class="text-3xl font-bold text-center text-gray-800 mb-6">Upload File for Prediction</h1>
        <form action="/predict" method="post" enctype="multipart/form-data" class="space-y-4">
            <div id="drop-area" class="drag-area">
                <p class="text-gray-600">Drag & Drop your file here<br>or click to select</p>
                <input type="file" id="fileInput" name="file" class="hidden" required>
                <button type="button" id="browseBtn" 
                        class="mt-3 px-4 py-2 bg-purple-600 text-white rounded-lg shadow hover:bg-purple-700">
                    Browse File
                </button>
                <p id="fileName" class="mt-2 text-sm text-gray-700"></p>
            </div>
            <button type="submit"
                    class="w-full bg-purple-600 text-white font-semibold py-2 px-4 rounded-lg shadow-md hover:bg-purple-700 transition-colors">
                Predict
            </button>
        </form>

        {% if prediction_result %}
            <div class="mt-6 p-4 bg-purple-100 text-purple-800 rounded-lg shadow-inner text-center font-bold">
                {{ prediction_result }}
            </div>
        {% endif %}

        <p class="mt-4 text-center text-gray-600">
            <a href="/dashboard" class="text-blue-600 hover:underline">Go to Dashboard</a>
        </p>
    </div>

    <script>
        const dropArea = document.getElementById("drop-area");
        const fileInput = document.getElementById("fileInput");
        const browseBtn = document.getElementById("browseBtn");
        const fileName = document.getElementById("fileName");

        // Open file dialog
        browseBtn.addEventListener("click", () => fileInput.click());

        // Show selected file name
        fileInput.addEventListener("change", () => {
            fileName.textContent = fileInput.files[0]?.name || "";
        });

        // Drag & Drop events
        dropArea.addEventListener("dragover", (e) => {
            e.preventDefault();
            dropArea.classList.add("dragover");
        });

        dropArea.addEventListener("dragleave", () => {
            dropArea.classList.remove("dragover");
        });

        dropArea.addEventListener("drop", (e) => {
            e.preventDefault();
            dropArea.classList.remove("dragover");

            if (e.dataTransfer.files.length > 0) {
                fileInput.files = e.dataTransfer.files;
                fileName.textContent = e.dataTransfer.files[0].name;
            }
        });
    </script>
</body>
</html>

"""
}

if __name__ == "__main__":
    app.run(debug=True)
