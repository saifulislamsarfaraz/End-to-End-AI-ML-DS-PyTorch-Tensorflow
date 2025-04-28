from flask import Flask

app = Flask(__name__)

@app.route("/")
def welcome():
    return "Welcome to this Flask course. This should be an amazing course. this is one thing "
@app.route("/index")
def index():
    return "Welcome to the index page.Alhamdulillah I should need to time change so that it could be great if any change is happend"

if __name__ == "__main__":
    app.run(debug=True)