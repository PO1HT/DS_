from flask import Flask, request, render_template
import tensorflow as tf
import pickle
app = Flask(__name__)

@app.route('/')
def choose_prediction_method():
    return render_template('main.html')

def durability(params):
    # Из дампа загружаем нормализатор для входных / выходных данных
    # из дампа загружаем модель расчета
    scaler_in_durable = pickle.load(open('models/scale_model_durable.pkl', 'rb'))
    model = pickle.load(open('models/model_durable.pkl', 'rb'))
    scaler_out_durable1 = pickle.load(open('models/scaler_out_durable1.pkl', 'rb'))
    scaler_out_durable2 = pickle.load(open('models/scaler_out_durable2.pkl', 'rb'))
    predict_ = model.predict(scaler_in_durable.transform([params]))
    predict_result = predict_ * scaler_out_durable1 + scaler_out_durable2
    return predict_result

def elasticity(params):

    scaler_in_elasticity = pickle.load(open('models/scaler_model_regr_elasticity.pkl', 'rb'))
    model = pickle.load(open('models/model_regr_elasticity.pkl', 'rb'))
    scaler_out_elasticity1 = pickle.load(open('models/scaler_model_regr_elasticity1.pkl', 'rb'))
    scaler_out_elasticity2 = pickle.load(open('models/scaler_model_regr_elasticity2.pkl', 'rb'))
    predict_ = model.predict(scaler_in_elasticity.transform([params]))
    predict_result = predict_ * scaler_out_elasticity1 + scaler_out_elasticity2
    return predict_result

def matrix_filler(params):
    scaler_in_matrix_filler = pickle.load(open('models/scaler_in_matrix_filler.pkl', 'rb'))
    model = pickle.load(open('models/model_k_matrix_filler.pkl', 'rb'))
    scaler_out_matrix_filler1 = pickle.load(open('models/scaler_out_matrix_filler1.pkl', 'rb'))
    scaler_out_matrix_filler2 = pickle.load(open('models/scaler_out_matrix_filler2.pkl', 'rb'))
    predict_ = model.predict(scaler_in_matrix_filler.transform([params]))
    predict_result = predict_ * scaler_out_matrix_filler1 + scaler_out_matrix_filler2
    return predict_result


@app.route('/durable/', methods=['POST', 'GET'])
def endpiont_durability():
    msg = ''
    if request.method == 'POST':
        param_list = ('mn', 'plot', 'mup', 'ko', 'seg', 'tv', 'pp', 'pr', 'ps', 'shn', 'pln')
        params = []
        for i in param_list:
            param = request.form.get(i)
            params.append(param)
        params.append('4,3')
        params = [float(i.replace(',', '.')) for i in params]

        msg = f'Модуля упругости: {durability(params)} ГПа'
    return render_template('elasticity.html', message=msg)

@app.route('/elasticity/', methods=['POST', 'GET'])
def endpoint_elasticity():
    msg = ''
    if request.method == 'POST':
        param_list = ('mn', 'plot', 'mup', 'ko', 'seg', 'tv', 'pp', 'mupr', 'ps', 'shn', 'pln')
        params = []
        for i in param_list:
            param = request.form.get(i)
            params.append(param)
        params.append('4,3')
        params = [float(i.replace(',', '.')) for i in params]

        msg = f'Прочность при растяжении: {elasticity(params)} МПа'
    return render_template('durability.html', message=msg)

@app.route('/matrix/', methods=['POST', 'GET'])
def endpoint_matrix_filler():
    msg = ''
    if request.method == 'POST':
        param_list = ('plot', 'mup', 'ko', 'seg', 'tv', 'pp', 'mupr', 'pr', 'ps', 'shn', 'pln')
        params = []
        for i in param_list:
            param = request.form.get(i)
            params.append(param)
        params = [float(i.replace(',', '.')) for i in params]
        params.append('4,3')
        msg = f'Матрица-наполнитель: {matrix_filler(params)}'
    return render_template('matrix_filler.html', message=msg)

if __name__ == '__main__':
    app.debug = True
    app.run()
