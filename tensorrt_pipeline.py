# model.export(format = 'engine', half = True)

def tensorrt_model(model):
    return model.export(format = 'engine', half = True)