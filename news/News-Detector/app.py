from flask import Flask, escape, request, render_template
import pickle
model = pickle.load(open("News-Detector\Final_model.pkl",'rb'))
vector = pickle.load(open("News-Detector\Rizor.pkl",'rb'))

#initialize Tfidfvectorizer

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("Home.html")
@app.route('/Prediction')
def Prediction():
    return render_template("index.html")
@app.route('/contact')
def contact():
    return render_template("contact.html")
@app.route('/about')
def about():
    return render_template("about.html")
@app.route('/home')
def Home():
    return render_template("home.html")
@app.route('/Check',methods=['GET', 'POST'])
def predict():
    if request.method == "POST":
        news = str(request.form['news'])
        print(news)
        pred = model.predict(vector.transform([news]))[0]
        print(pred)
        with open("Predict.txt",'w') as f:
            f.write(pred)
       
        if pred=="REAL":
            return render_template("index.html",prediction_text="Accurate")
        elif pred=="FAKE":
            return render_template("index.html",prediction_text="InAccurate")
    else:
        return render_template("index.html")
if __name__=='__main__':
    app.debug = True
    app.run()