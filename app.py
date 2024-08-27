import os
from flask import Flask

from app1.gradient_descent import create_app as create_app1
from app2.gradient_descent_with_momentum import create_app as create_app2
from app3.support_vector_machines import create_app as create_app3
from app4.feedforward_neural_network import create_app as create_app4

server = Flask(__name__)

app1 = create_app1()
app1.init_app(server)

app2 = create_app2()
app2.init_app(server)

app3 = create_app3()
app3.init_app(server)

app4 = create_app4()
app4.init_app(server)

if __name__ == "__main__":
    server.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8050)))