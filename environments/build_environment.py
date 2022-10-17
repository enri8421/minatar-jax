from importlib import import_module






def get(env_name, **args):
    factory = getattr(import_module("environments."+env_name), env_name)
    env = factory(**args)
    return env





    


