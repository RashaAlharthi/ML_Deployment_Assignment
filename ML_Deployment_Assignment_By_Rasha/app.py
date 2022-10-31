from flask import Flask, request, render_template
import pandas as pd
import pickle

# Giving our application a name --> app
app = Flask(__name__)

# Loading our model and encoder 
encoder = pickle.load(open('notebook/encoder.pkl', 'rb'))
scaler = pickle.load(open('notebook/scaler.pkl', 'rb'))
model = pickle.load(open('notebook/model.pkl', 'rb'))

# This will take you to the home HTML page --> in this example it is called home.html
# Note: in general applications home pages are usually called --> index.html
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():

    # Geting features and values from the usern 
    data = request.form.to_dict()
    str_features = {}
    num_features = {}
    
    # Separate strings for hot encoding and numners to convert to integer
    for k, v in data.items():
        if v.isdigit():
            num_features[k] = [int(v)] 
        else:
            str_features[k] = [v]
            
    # Creating dataframes to pass them to encoder and model 
    # Important Note: This is how they were trained in jupyter notebook
    df1 = pd.DataFrame(data=str_features)  
    df2 = pd.DataFrame(data=num_features)

    # Encode strings
#    enc_arr = encoder.transform(df1)    
    cols = df1.columns.tolist()
    for column in cols:
        df1[column] = encoder.fit_transform(df1[column])
    
    # Add encoded values to dataframe
#    df2[encoder.get_feature_names()] = enc_arr.toarray()
    df1 = df1.join(df2)
    
    # Make predictions on dataframe
    prediction = model.predict(df1)
    # get the output
    output = prediction[0]

    return render_template('predictions.html', prediction= output, data=data)

if __name__ == "__main__":
    app.run(debug=True)