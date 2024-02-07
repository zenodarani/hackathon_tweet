from flask import Flask, render_template, request, redirect, url_for, jsonify
import pandas as pd
import io
import emoji
from supervised_part.functions import model


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def predict_res(data):
    return model(data)

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    df = pd.read_excel(file)
    print(df.columns)
    
    predictions = predict_res(df)

    # Add a new column 'Row_Index' to the DataFrame starting from 1
    df['Row_Index'] = range(1, len(df) + 1)

    # Ensure at least one label is true for each row
    df['label_1'] = False
    df['label_2'] = False
    df['label_3'] = False

    for i, predict in enumerate(predictions):
        # Set label to True only if it's not already True
        if predict == 'PP' and not df.at[i, 'label_1']:
            df.at[i, 'label_1'] = True
        elif predict == 'PO' and not df.at[i, 'label_2']:
            df.at[i, 'label_2'] = True
        elif predict == 'UN' and not df.at[i, 'label_3']:
            df.at[i, 'label_3'] = True

    labels = [{'row_index': i, 'label_1': label_1, 'label_2': label_2, 'label_3': label_3}
              for i, (label_1, label_2, label_3) in enumerate(zip(df['label_1'], df['label_2'], df['label_3']))]

    return render_template('label.html', data=zip(df.iloc[:, 0], df['Row_Index'], labels), emoji=emoji)



@app.route('/export', methods=['POST'])
def export():
    labeled_data = []
    text_data = {}

    for key, value in request.form.items():
        if '_hidden_text' in key:
            row_index = int(key.split('_')[0])
            text_data[row_index] = value

        if '_label' in key:
            row_index = int(key.split('_')[0])
            labeled_data.append({'text': text_data.get(row_index, None), 'value': value})

    export_df = pd.DataFrame(labeled_data)

    export_df.to_excel('labeled_data.xlsx', index=False)

    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify({'status': 'success', 'message': 'Export successful'})
    else:
        return redirect(url_for('index'))




if __name__ == '__main__':
    app.run(debug=True)