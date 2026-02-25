import torch 
from torch import nn

#   Validamos control sobre la GPU
device = "cuda" if torch.cuda.is_available()else "cpu"
print(f"Dispositivo de entrenamiento asignado: {device}\n")

#   clase de la nn
class NeuronalNetworkTesis(nn.Module):
    def __init__(self):
        super().__init__()
        #   definicion de "capas" del cerebro
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_features= 512, out_features=256), #capa de entrada a co1(capa oculta=co)
            nn.ReLU(),
            nn.Linear(in_features=256, out_features= 128), #co1 a co2
            nn.ReLU(),
            nn.Linear(in_features=128, out_features= 10) #co2 a salida (ej.10clases)
        )

    #   metodo forward define como fluyen los datos(hacia adelante)
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
#   Instanciar el modelo y enviarlo a GPU
modelo = NeuronalNetworkTesis().to(device)

print("Estructuctura de la NN:\n")
print(modelo)

print("\n --- INICIANDO FORWARD PASS TEST ---")

#   Datos falsos simulando la entrada
#   Definimos un tesor aleatorio de una fila(un dato) y 512 columnas(features)
#   Trabajamos directamente con GPU donde vive el modelo .to(device)
X= torch.rand(1,512, device = device)
print(f"Forma de los datos de entrada: {X.shape}")
#   Al pasar los datos al modelo invocamos automaticamente la f de 'forward'
logits = modelo(X)
#   Mostramos el resultado (salida capeada en 10 features/ultima capa)
print(f"Logits (Resultado crudo): {logits}")

#   Prediccion final
#   Softmax convierte esos numeros crudos en % de probabilidad
probabilities = nn.Softmax(dim=1)(logits)
final_prediction = probabilities.argmax(1)

print(f"Probabilidades: {probabilities}")
print(f"La red predice que este dato falso pertenece a la clase: {final_prediction}")