from flask import Flask, request, render_template_string
import joblib
import numpy as np

app = Flask(__name__)

# Load model and encoders
model = joblib.load('model.h5')
label_encoders = joblib.load('label_encoders.pkl')
crime_columns = joblib.load('crime_columns.pkl')

# Lists
states = [
    'A & N ISLANDS', 'ANDHRA PRADESH', 'ARUNACHAL PRADESH', 'ASSAM', 'BIHAR', 'CHANDIGARH', 'CHHATTISGARH',
    'D & N HAVELI', 'DAMAN & DIU', 'DELHI', 'GOA', 'GUJARAT', 'HARYANA', 'HIMACHAL PRADESH', 'JAMMU & KASHMIR',
    'JHARKHAND', 'KARNATAKA', 'KERALA', 'LAKSHADWEEP', 'MADHYA PRADESH', 'MAHARASHTRA', 'MANIPUR', 'MEGHALAYA',
    'MIZORAM', 'NAGALAND', 'ODISHA', 'PUDUCHERRY', 'PUNJAB', 'RAJASTHAN', 'SIKKIM', 'TAMIL NADU', 'TELANGANA',
    'TRIPURA', 'UTTAR PRADESH', 'UTTARAKHAND', 'WEST BENGAL'
]
years = list(range(2001, 2022))
crime_types = ['Rape', 'Kidnap', 'Dowry Deaths', 'Assaults', 'Assualts against modesty', 'Domestic Violence', 'Women Trafficking']

# HTML template
template = '''
<!doctype html>
<html lang="en">
<head>
    <title>Crime Arrest Prediction</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            background-color: #f0f2f5; 
            display: flex; 
            justify-content: center; 
            align-items: center; 
            height: 100vh;
            margin: 0;
        }
        .container {
            background: white;
            padding: 40px;
            border-radius: 15px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            width: 450px;
            text-align: center;
        }
        input, select, button {
            padding: 8px;
            margin: 6px 0;
            width: 100%;
            border-radius: 5px;
            border: 1px solid #ccc;
            box-sizing: border-box;
        }
        button {
            background-color: #007BFF;
            color: white;
            border: none;
            cursor: pointer;
            margin-top: 15px;
        }
        button:hover {
            background-color: #0056b3;
        }
        h1 {
            color: #333;
            margin-bottom: 20px;
        }
        .result {
            font-size: 22px;
            margin-top: 20px;
            color: green;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }
        td {
            padding: 5px 8px;
        }
        label {
            font-weight: bold;
            font-size: 14px;
        }
        .crime-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }
        .crime-row input {
            width: 45%;
        }
        .crime-row label {
            width: 50%;
            text-align: left;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚓 Crime Arrest Prediction</h1>
        <form method="POST" action="/predict">
            <select name="state" required>
                {% for state in states %}
                    <option value="{{ state }}">{{ state }}</option>
                {% endfor %}
            </select>

            <select name="year" required>
                {% for year in years %}
                    <option value="{{ year }}">{{ year }}</option>
                {% endfor %}
            </select>

            {% for crime in crime_types %}
            <div class="crime-row">
                <label>{{ crime }}:</label>
                <input type="number" name="{{ crime }}" min="0" max="9999" required>
            </div>
            {% endfor %}

            <button type="submit">Predict</button>
        </form>

        {% if prediction %}
            <div class="result">
                <p><strong>Prediction: {{ prediction }}</strong></p>
            </div>
        {% endif %}
    </div>
</body>
</html>
'''


@app.route('/', methods=['GET'])
def home():
    return render_template_string(template, states=states, years=years, crime_types=crime_types, prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    # Form data
    state = request.form['state']
    year = request.form['year']

    # Only use first crime column (default, like 'Rape') to predict
    selected_crime = crime_types[0]  # e.g., 'Rape'
    no_of_cases = int(request.form[selected_crime])

    # Encoding
    state_encoded = label_encoders['State'].transform([state])[0]
    year_encoded = label_encoders['Year'].transform([str(year)])[0]

    # Input features
    input_features = np.zeros((1, 3))
    input_features[0, 0] = state_encoded
    input_features[0, 1] = year_encoded
    input_features[0, 2] = no_of_cases

    # Prediction
    prediction_raw = model.predict(input_features)[0]
    prediction = 'Arrested' if prediction_raw == 'Yes' else 'Not Arrested'

    return render_template_string(template, states=states, years=years, crime_types=crime_types, prediction=prediction)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
