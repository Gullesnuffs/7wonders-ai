import card_database
import copy
import random


class State:

    def __init__(self, players):
        self.numPlayers = players
        self.players = []
        for i in range(self.numPlayers):
            self.players.append(Player(Bot()))

        self.initAge(1)

    def initAge(self, age):
        self.age = age
        cards = card_database.getCards(age=self.age, players=len(self.players))
        random.shuffle(cards)
        for i in range(self.numPlayers):
            self.players[i].hand = cards[i * 7:(i + 1) * 7]

    def getHiddenState(self, perspective):
        state = copy.copy(self)
        state.players = [self.players[perspective]]
        for i in range(perspective + 1, perspective + state.numPlayers):
            state.players.append(self.players[i].convertToHidden())
        return state

    def performMoves(self, moves):
        state = copy.deepcopy(self)
        oldHands = []
        for i in range(state.numPlayers):
            state.players[i].performMove(moves[i])
            oldHands.append(state.players[i].hand)
        for i in range(state.numPlayers):
            if (age == 2):
                state.players[i].hand = oldHands[i + 1]
            else:
                state.players[i].hand = oldHands[i - 1]
        return state

    def print(self):
        for player in self.players:
            player.print()


class Player:

    def __init__(self, bot):
        self.bot = bot

    def print(self):
        print('Bought cards')
        for card in self.boughtCards:
            card.print()
        print('\nHand')
        for card in self.hand:
            card.print()
        print('\n')

    def convertToHidden(self):
        player = copy.copy(self)
        player.hand = []
        return player

    def performMove(self, card):
        self.boughtCards.append(card)
        self.removeCardFromHand(card)

    def removeCardFromHand(self, card):
        for i in range(len(self.hand)):
            if (self.hand[i] == card):
                self.hand[i] = self.hand[-1]
                self.hand.pop()
                break


class Bot:

    def __init__(self):
        print("Init bot")

    def getMove(state):
        return random.choice(state.players[0].hand)


def playGame(players=3):
    state = State(players=players)
    for cardsLeft in range(7, 1, -1):
        moves = []
        for i in range(players):
            player = state.players[i]
            moves.append(player.bot.getMove(state.getHiddenState(i)))
        state = state.performMoves(moves)
        state.print()


print('Play game')
playGame()
