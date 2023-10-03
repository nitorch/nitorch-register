class MissingImport:

    def __init__(self, error):
        self.error = error

    def __getattribute__(self, name):
        raise self.error

    def __setattr__(self, name, value):
        raise self.error

    def __bool__(self):
        return False
