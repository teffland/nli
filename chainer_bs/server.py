import argparse
import os
import os.path as osp

from flask import Flask, render_template
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html', 
                           experiment_dir=experiment_dir)

@app.route('/home')
def experiments_home():
    experiments = []
    for name in os.listdir(experiment_dir):
        experiments.append({
            'name':name
        })
    return render_template('experiments_home.html',
                           experiments=experiments)

@app.route('/experiment/<name>')
def experiment_details(name):
    return name

    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--experiment_dir', type=str, required=True,
                        help='Path to the directory containing the experiments')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    globals()['experiment_dir'] = args.experiment_dir
    app.run(host="localhost", port=5000, debug=True)
