from flask import Flask, render_template, request
import pickle
import numpy as np

# load model
with open('./data_science_assin_NB_model.pkl','rb') as f :
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        gender = int(request.form.get('gender'))
        age = int(request.form.get('age'))
        salary = int(request.form.get('salary'))
        
        print(gender, age, salary)
        arr = np.array([[gender, age, salary]])
            
        result = model.predict(arr)
        print(result)
    
        return render_template('result.html', data=result)
    else:
        return render_template('home.html')

if __name__ == "__main__":
    app.run(debug=True)

