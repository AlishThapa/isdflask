from flask import Flask, render_template, request
import sqlite3
import pandas as pd
import math
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import pickle
import joblib
app = Flask(__name__)

# Function to load CSV data into SQLite database
def load_csv_to_db():
    con = sqlite3.connect('alishDb.db')
    df = pd.read_csv('isd.csv')
    df.to_sql('insider', con, if_exists='replace', index=False)
    con.close()

@app.route('/load-csv')
def load_csv():
    load_csv_to_db()
    return "CSV data loaded into SQLite database successfully!"

def load_newcsv_to_db():
    con = sqlite3.connect('newDb.db')
    df = pd.read_csv('X_testss.csv')
    df.to_sql('insider', con, if_exists='replace', index=False)
    con.close()

@app.route('/load-new-csv')
def load_new_csv():
    load_newcsv_to_db()
    return "CSV data loaded into SQLite database successfully!"

@app.route('/')
def login():
    return render_template('login.html')


@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')



@app.route('/insider-data')
def show_data():
    page = request.args.get('page', 1, type=int)
    rows_per_page = 100  # Adjust based on your preference and performance testing
    offset = (page - 1) * rows_per_page
    
    con = sqlite3.connect('alishDb.db')
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    
    # Fetch the total number of rows for pagination
    cur.execute("SELECT COUNT(*) AS total FROM insider")
    total_rows = cur.fetchone()['total']
    total_pages = (total_rows // rows_per_page) + (total_rows % rows_per_page > 0)
    
    # Dynamically fetch rows with limit and offset for pagination
    cur.execute(f"SELECT * FROM insider LIMIT {rows_per_page} OFFSET {offset}")
    rows = cur.fetchall()
    
    # Assuming the first row is representative of the structure
    if rows:
        columns = rows[0].keys()
    else:
        columns = []
    
    data = [dict(row) for row in rows]
    con.close()
    
    return render_template('show_data.html', columns=columns, data=data, page=page, total_pages=total_pages)


@app.route('/dashboard')
def dashboard():
    con = sqlite3.connect('newDb.db')
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute("SELECT DISTINCT \"0\" FROM insider")  # Adjust the query to select only the "0" column
    user_ids = [row['0'] for row in cur.fetchall()]
    con.close()
    return render_template('dashboard.html', users=user_ids)


@app.route('/fetch-user-data/<user_id>')
def fetch_user_data(user_id):
    con = sqlite3.connect('newDb.db')
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    #from X_testss.csv i want to show the number in 3rd column in the dropdown list which i indicate as a users.generate a code to perform that
    # Dynamically fetching the column names.
    cur.execute("SELECT * FROM insider WHERE  \"0\" = ?", (user_id,))
    row = cur.fetchone()
    if not row:
        return "No data found for user."

    # Generate headers based on column names
    headers = row.keys()
    all_rows = cur.fetchall()  # Fetch all rows for the user_id

    table_html = '<table class="data-table"><tr>'  # Add a class to the table
    for header in headers:
        table_html += f'<th>{header}</th>'
    table_html += '</tr>'
    # Adding the first row data
    table_html += "<tr>"
    for value in row:
        table_html += f"<td>{value}</td>"
    table_html += "</tr>"

    # Adding the rest of the rows data
    for row in all_rows:
        table_html += "<tr>"
        for value in row:
            table_html += f"<td>{value}</td>"
        table_html += "</tr>"
    
    table_html += "</table>"
    return table_html


@app.route('/analyze')
def analyze():
    user_id = request.args.get('userId')
    return render_template('analyze.html', user_id=user_id)
    


# def load_model(path='RandomForestModel.pkl'):
#     model = joblib.load(path)
#     return model


def load_test_data(user_id, x_test_path='X_testss.csv', y_test_path='Ytest.csv'):
    X_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path)
    matching_rows = X_test[X_test['0'] == user_id]
    if not matching_rows.empty:
        X_test = matching_rows
        y_test = y_test
    else:
        X_test = pd.DataFrame()
        y_test = pd.Series()
    return X_test, y_test.to_numpy()

test_data = pd.read_csv('X_testss.csv')

with open('FinalRandomForest.pkl', 'rb') as file:
    modelrf = pickle.load(file)
with open('FinalDecisionTree.pkl', 'rb') as file:
    modeldt = pickle.load(file)


@app.route('/result', methods=['GET'])
def result():
    user_id = request.args.get('userId')
    model_name = request.args.get('model')

    if user_id:
        user_id = math.floor(float(user_id))  # Convert to integer by rounding down
        X_test, y_test = load_test_data(user_id)
        
        if not X_test.empty:
            if model_name == 'DecisionTree':
                model = modeldt
            elif model_name == 'RandomForest':
                model = modelrf
            else:
                # No prediction for 'mlpClassifier'
                return render_template('result.html', user_id=user_id, predictions=[],GroundTruth =[])
            predictions = model.predict(X_test)
            print(type(predictions),type(y_test))
            return render_template('result.html', user_id=user_id, predictions=predictions,GroundTruth =predictions)
        else:
            return "No data found for the provided user ID."
    else:
        return "User ID is required."


if __name__ == '__main__':
    app.run(debug=True)













@app.route('/confusion-matrix')
def show_confusion_matrix():
    # Load the model
    model = load_model() 
    # Load or generate your X_test and y_test here
    X_test, y_test = load_test_data()  
    # Predict using the loaded model
    y_pred = model.predict(X_test)
    # Generate the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    # Plotting the confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')   
    # Convert plot to PNG image
    pngImage = BytesIO()
    plt.savefig(pngImage, format='png')
    plt.close(fig)
    pngImage.seek(0)    
    # Encode PNG image to base64 string
    imageBase64 = base64.b64encode(pngImage.getvalue()).decode('utf8')
    # Render the confusion matrix HTML
    return render_template('confusion_matrix.html', imageBase64=imageBase64)


@app.route('/confusion-matrix-image')
def get_confusion_matrix_image():  
    # Load the model
    model = load_model()
    # Load or generate your X_test and y_test here
    X_test, y_test = load_test_data()
    # Predict using the loaded model
    y_pred = model.predict(X_test)
    # Generate the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    # Plotting the confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    # Convert plot to PNG image
    pngImage = BytesIO()
    plt.savefig(pngImage, format='png')
    plt.close(fig)
    pngImage.seek(0)
    # Encode PNG image to base64 string
    imageBase64 = base64.b64encode(pngImage.getvalue()).decode('utf8')
    return imageBase64