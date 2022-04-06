"""
FUNCIONES VARIAS
Cálculo de tamaño de capas, etc...
"""
def pyramid_rule(h_layers:int, input_size:int, output_size:int) -> list:
    """
    Devuelve una lista con el tamaño de las capas para una red neuronal, siguiendo la regla de la piramide geometrica.\n
    h_layers: numero de capas ocultas deseadas\n
    input_size: tamaño de la capa de entrada\n
    output_size: tamaño de la capa de salida
    """
    layers = []
    if h_layers < 1:
        print("No layers")
        return []
    print("Layers for input %d and output %d:" % (input_size,  output_size))
    rate = (input_size/output_size)**(1/(h_layers+1))
    for l in range(h_layers):
        layer_size = output_size*(rate**(h_layers-l))
        layer_size = round(layer_size)
        layers.append(layer_size)
        print("Layer %d: %d neurons" % (l+1, layer_size))
    return layers