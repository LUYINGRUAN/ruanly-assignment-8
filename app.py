from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from logistic_regression import do_experiments

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Define the main route to serve the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle experiment parameters and trigger the experiment
@app.route('/run_experiment', methods=['POST'])
def run_experiment():
    # Get data from the request
    data = request.json
    start = float(data.get('start'))
    end = float(data.get('end'))
    step_num = int(data.get('step_num'))

    # Run the experiment
    do_experiments(start, end, step_num)

    # Paths to result images
    dataset_img = "results/dataset.png"
    parameters_img = "results/parameters_vs_shift_distance.png"    

    # Check if images exist and return their paths
    return jsonify({
        "dataset_img": dataset_img if os.path.exists(dataset_img) else None,
        "parameters_img": parameters_img if os.path.exists(parameters_img) else None
    })

# Route to serve result images
@app.route('/results/<filename>')
def results(filename):
    return send_from_directory('results', filename)

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True)
