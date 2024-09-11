import os
from flask import Flask

from app0.gradient_descent import create_app as create_app0
from app1.gradient_descent_with_momentum import create_app as create_app1
from app2.stochastic_gradient_descent import create_app as create_app2
from app3.support_vector_machines import create_app as create_app3
from app4.feedforward_neural_network import create_app as create_app4
from app6.tensorboard_window import create_app as create_app6

server = Flask(__name__)

app0 = create_app0()
app0.init_app(server)

app1 = create_app1()
app1.init_app(server)

app2 = create_app2()
app2.init_app(server)

app3 = create_app3()
app3.init_app(server)

app4 = create_app4()
app4.init_app(server)

app6 = create_app4()
app6.init_app(server)
server.register_blueprint(app6, url_prefix='/tensorboard_view')

if __name__ == "__main__":
    server.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8050)))