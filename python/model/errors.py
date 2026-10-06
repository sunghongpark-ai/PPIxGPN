class PPIxGPNError(ValueError):
    def __init__(self, identifier, message):
        self.identifier = identifier
        self.message = message
        super().__init__(f"{identifier}: {message}")
