from flask import Flask, render_template
from flask import request
import joblib
linearmodel=joblib.load(open('linear.pkl','rb'))
app=Flask(__name__,template_folder='template')
@app.route('/')
def home():
     return render_template('flaskdummy.html')

@app.route('/predict',methods=['POST','GET'])
def prediction():
    if request.method == 'POST':
        g=request.args.get('gender')
        o=request.args.get('own')
        l=request.args.get('location')
        s=request.args.get('status')
        sa=request.args.get('salary')
        a=request.args.get('age')
        c=request.args.get('cat')
        ch=request.args.get('child')
        b=request.args.get('buy')
    
        test = [g,o,l,s,sa,a,c,ch,b]
        predicted = linearmodel.predict(test)
        return render_template('flaskdummy.html',pred="Probability of depression is {}".format(predicted))


if __name__== '__main__' :
    app.run(port="5000",debug=True)