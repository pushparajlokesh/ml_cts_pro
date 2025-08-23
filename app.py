from flask import Flask, render_template, request, redirect, session, send_file
import mysql.connector
from flask_bcrypt import Bcrypt
import pickle
import pandas as pd
import os

app = Flask(__name__)
app.secret_key = "loke2005@."
bcrypt = Bcrypt(app)

# Load your ML model (pickle file)
   # <-- put your model file name here
model = pickle.load(open("lgbm_multioutput_model.pkl", "rb"))


# MySQL connection
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="loke2005@.",
        database="myappdb"
    )

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        password = bcrypt.generate_password_hash(request.form["password"]).decode("utf-8")

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (username, email, password) VALUES (%s, %s, %s)",
                       (username, email, password))
        conn.commit()
        cursor.close()
        conn.close()
        return redirect("/login")
    return render_template("signup.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE email=%s", (email,))
        user = cursor.fetchone()
        cursor.close()
        conn.close()

        if user and bcrypt.check_password_hash(user["password"], password):
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            return redirect("/dashboard")
        else:
            return "Invalid credentials"
    return render_template("login.html")

@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    if "user_id" not in session:
        return redirect("/login")

    result_preview = None

    if request.method == "POST":
        file = request.files["sqlfile"]
        if file:
            # Save uploaded SQL file
            filepath = os.path.join("uploads", file.filename)
            os.makedirs("uploads", exist_ok=True)
            file.save(filepath)

            # Read SQL file into DataFrame
            conn = get_db_connection()
            cursor = conn.cursor()
            sql_commands = open(filepath, "r").read()
            for result in cursor.execute(sql_commands, multi=True):
                pass
            conn.commit()

            # Example: Fetch data from a table (you can change "input_table")
            df = pd.read_sql("SELECT * FROM input_table", conn)

            # Run ML model prediction
            predictions = model.predict(df)  # works if df matches your model’s features
            df["Predictions"] = predictions

            # Save output to CSV
            output_csv = "output.csv"
            df.to_csv(output_csv, index=False)

            # Store results into MySQL table
            cursor.execute("DROP TABLE IF EXISTS model_output")
            cols = ", ".join([f"`{c}` TEXT" for c in df.columns])
            cursor.execute(f"CREATE TABLE model_output ({cols})")

            for _, row in df.iterrows():
                cursor.execute(
                    f"INSERT INTO model_output VALUES ({','.join(['%s']*len(row))})",
                    tuple(row)
                )
            conn.commit()
            cursor.close()
            conn.close()

            result_preview = df.head(10).to_html(classes="table table-bordered")

    return render_template("dashboard.html", username=session["username"], result=result_preview)

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")

if __name__ == "__main__":
    app.run(debug=True)
