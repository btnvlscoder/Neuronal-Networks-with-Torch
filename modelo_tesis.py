import torch 
from torch import nn

# Validamos control sobre la GPU
device = "cuda" if torch.cuda.is_available()else "cpu"
print(f"Dispositivo de entrenamiento asignado: {device}\n")

#clase de la nn
class NeuronalNetworkTesis(nn.Module):
    def __init__(self):
        super().__init__()
        #definicion de "capas" del cerebro
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_features= 512, out_features=256), #capa de entrada a co1(capa oculta=co)
            nn.ReLU(),
            nn.Linear(in_features=256, out_features= 128), #co1 a co2
            nn.ReLU(),
            nn.Linear(in_features=128, out_features= 10) #co2 a salida (ej.10clases)
        )

    #metodo forward define como fluyen los datos(hacia adelante)
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
# Instanciar el modelo y enviarlo a GPU
modelo = NeuronalNetworkTesis().to(device)

print("Estructuctura de la NN:\n")
print(modelo)


