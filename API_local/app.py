from flask import Flask, render_template, request, jsonify, send_file
#from werkzeug.utils import secure_filename
import joblib
import traceback
import pandas as pd
import os
import numpy as np
from flask_table import Table, Col

app = Flask(__name__)

@app.route('/upload')
def upload_file():
    return render_template('upload.html')

@app.route('/uploader', methods = ['POST'])
def uploader_file():
    f = request.files['upload_file']
    f.save(os.path.join(os.path.dirname(__file__), 'instance', 'app_upload', 'query.csv'))
    if request.method == 'POST':
        if logreg_model:
            try:
                query = pd.read_csv(r'instance/app_upload/query.csv')

                features = ['margin_low', 'length']
                query = query[features]
                query_col = query.columns
                query_scaled = std_scale_model.transform(query.values)
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

                prediction.to_csv(r'instance/app_output/predict.csv')


                return render_template('upload.html', table=table_html, name_file_uploaded=name_file_uploaded,
                                       count_true=count_true, count_false=count_false, upload_done=True)


            except:
                return jsonify({'trace': traceback.format_exc()})
        else:
            print('Train the model first')
            'No model available'

@app.route('/download_file')
def return_file():
    return send_file('instance/app_output/predict.csv',
                     attachment_filename='predict.csv',
                     as_attachment=True)

if __name__ == '__main__':
    logreg_model = joblib.load('models/log_reg.model')
    print('Model loaded')
    std_scale_model = joblib.load('models/std_scale.model')
    print('Scaler Model loaded')
    app.run(host='localhost', port=5000, debug=True)

