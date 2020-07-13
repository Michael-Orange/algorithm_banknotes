from flask import Flask, render_template, request, jsonify, send_file, request
from werkzeug.utils import secure_filename
import joblib
import traceback
import pandas as pd
import os
from flask_table import Table, Col
import numpy as np


app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = '/tmp'

def loading_models():
    logreg_model = joblib.load(os.path.join(os.path.dirname(__file__), 'models', 'log_reg.model'))
    std_scale_model = joblib.load(os.path.join(os.path.dirname(__file__), 'models','std_scale.model'))

    return logreg_model, std_scale_model

@app.route('/')
def upload_file():
    return render_template('upload.html')

@app.route('/uploader', methods = ['POST'])
def uploader_file():
    f = request.files.get('upload_file')
    f.save(os.path.join(app.config['UPLOAD_FOLDER'], 'query.csv'))

    if request.method == 'POST':
        
        logreg_model, std_scale_model = loading_models()
        
        if logreg_model:
            try:
                query = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], 'query.csv'))

                features = ['margin_low', 'length']
                query_features = query[features]
                query_col = query_features.columns
                query_scaled = std_scale_model.transform(query_features.values)
                query_scaled = pd.DataFrame(query_scaled, columns=query_col)

                prediction = pd.DataFrame(logreg_model.predict(query_scaled).transpose(), columns=['genuine'])
                prediction['prob_genuine(%)'] = np.round(logreg_model.predict_proba(query_scaled[features])[:, 1] * 100, 2)
                prediction = query.merge(prediction, left_index=True, right_index=True)

                count_true = prediction.loc[prediction['genuine'] == True]['genuine'].count()
                count_true = 'Number of banknotes predicted True/Genuine : {}'.format(count_true)

                count_false = prediction.loc[prediction['genuine'] == False]['genuine'].count()
                count_false = 'Number of banknotes predicted False/Counterfeit : {}'.format(count_false)

                name_file_uploaded = 'File uploaded: {}'.format(f.filename)
                table_html = prediction.to_html()

                prediction.to_csv(os.path.join(app.config['UPLOAD_FOLDER'], 'predict.csv'))

                return render_template('upload.html', table=table_html, name_file_uploaded=name_file_uploaded,
                                       count_true=count_true, count_false=count_false, upload_done=True)


            except:
                return jsonify({'trace': traceback.format_exc()})
        else:
            print('Train the model first')
            'No model available'

@app.route('/download_file')
def return_file():
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], 'predict.csv'), attachment_filename='predict.csv', as_attachment=True)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)