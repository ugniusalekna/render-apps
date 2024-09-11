import os
import subprocess
from flask import Blueprint, render_template_string

TENSORBOARD_LOGDIR = 'logs/'

def create_app():
    bp = Blueprint('tensorboard', __name__)

    @bp.route('/tensorboard')
    def tensorboard():
        subprocess.Popen(['tensorboard', '--logdir', TENSORBOARD_LOGDIR, '--host', '0.0.0.0', '--port', '6006'])

        tensorboard_template = '''
        <h1>TensorBoard Dashboard</h1>
        <iframe src="http://localhost:6006" width="100%" height="800px"></iframe>
        '''
        return render_template_string(tensorboard_template)

    return bp
