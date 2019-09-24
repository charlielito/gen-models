class Callbacks:
    def __init__(self, callbacks):
        self.callbacks = callbacks

    def __getattr__(self, name):
        def method(*args, **kwargs):
            for callback in self.callbacks:
                getattr(callback, name)(*args, **kwargs)

        return method
