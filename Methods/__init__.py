from Methods.method import Method


def get_prediction_method(method_name):
    if method_name == "SDL":
        from Methods.SDL import sdl_pytorch
        return Method("SDL", sdl_pytorch.train, sdl_pytorch.test, sdl_pytorch.update, {"epochs": 200, "early_stop": 10})
    elif method_name == "CAMARGO":
        from Methods.Camargo import adapter as camargo
        return Method("Camargo", camargo.train, camargo.test, camargo.update, {"epochs": 200, "early_stop": 10})
    elif method_name == "DIMAURO":
        from Methods.DiMauro import adapter as dimauro
        return Method("Di Mauro", dimauro.train, dimauro.test, dimauro.update, {"epochs": 200, "early_stop": 10})
    elif method_name == "TAX":
        from Methods.Tax import adapter as tax
        return Method("Tax", tax.train, tax.test, tax.update, {"epochs": 200, "early_stop": 10})
    else:
        print("ERROR: method name not found!")
        raise NotImplementedError()