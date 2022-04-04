import importlib

def import_string_to_class(import_string):
    """Parse an import string and return the class"""
    parts = import_string.split('.')
    module_name = '.'.join(parts[:len(parts) - 1])
    class_name = parts[-1]

    module = importlib.import_module(module_name)
    my_class = getattr(module, class_name)

    return my_class