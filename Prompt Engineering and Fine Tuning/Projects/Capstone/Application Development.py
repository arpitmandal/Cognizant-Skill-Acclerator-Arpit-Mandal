from flask import Flask, request, render_template
from transformers import pipeline
app = Flask(__name__)
model = pipeline("text-generation", model="../models/fine_tuned_model")

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        input_text = request.form["input"]
        output = model(input_text, max_length=50)[0]["generated_text"]
        return render_template("index.html", result=output)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
