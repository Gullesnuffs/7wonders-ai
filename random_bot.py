import random
from card import Move


class RandomBot:

    def __init__(self):
        self.name = "Random bot"

    def getBestPayOption(self, payOptions):
        best = None
        for payOption in payOptions:
            if (best is None or payOption.totalCost() < bestSum):
                bestSum = payOption.totalCost()
                best = payOption
        return best

    def getMove(self, state):
        player = state.players[0]
        playableCards = player.getPlayableCards()
        random.shuffle(playableCards)
        payOptions = []
        for i in range(len(playableCards)):
            card = playableCards[i]
            payOptions.append(state.getCardPayOptions(0, card))
            bestPayOption = self.getBestPayOption(payOptions[i])
            if bestPayOption is None:
                continue
            if (bestPayOption.totalCost() <= player.gold):
                return Move(card=card, payOption=bestPayOption)
        return Move(card=random.choice(player.hand), discard=True)

    def getMoves(self, states):
        return [self.getMove(state) for state in states]

    def train(self, state):
        pass
