class Move:
    
    def __init__(self, card, payOption = PayOption(), discard = False, buildWonder = False):
        self.card = card
        self.payOption = payOption
        self.discard = discard
        self.buildWonder = buildWonder
