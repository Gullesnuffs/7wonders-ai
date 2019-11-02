from dnn_reference_bot import DNNReferenceBot
from card_database import getCards, PURPLE_CARDS, ALL_WONDERS
from control import State
from card import PayOption, Move

PRINT = True

def getBool(query):
    while True:
        val = input(query)
        if val.lower() in {'y', 'yes', '"yes"'}:
            return True
        if val.lower() in {'n', 'no', '"no"'}:
            return False
        print('Write "yes" or "no"')


def getNumber(query):
    while True:
        val = input(query)
        try:
            return int(val)
        except:
            print('Not a valid number: "%s"' % val)

def getAutocompleteResult(query, considerationList, isWonder = False):
    typeName = 'wonder' if isWonder else 'card'
    while True:
        name = input(query)
        candidates = []
        for item in set(considerationList):
            if item.name.lower().startswith(name.lower()):
                candidates.append(item)
        if len(candidates) == 0:
            print('Invalid %s: "%s"' % (typeName, name))
            continue
        if len(candidates) == 1:
            return candidates[0]
        for i in range(len(candidates)):
            print('(%d): %s' % (i+1, candidates[i].name))
        print('(%d): Another %s' % (len(candidates)+1, typeName))
        while True:
            ind = getNumber('Enter number between 1 and %d: ' % (len(candidates)+1))
            if ind >= 1 and ind <= len(candidates):
                return candidates[ind-1]
            elif ind == len(candidates)+1:
                break

def getWonder(query):
    return getAutocompleteResult(query, ALL_WONDERS, isWonder = True)

def getCard(query, age, players):
    considerCards = getCards(age=age, players=players)
    if age == 3:
        considerCards += PURPLE_CARDS
    return getAutocompleteResult(query, considerCards)

def getHand(handSize, age, players):
    hand = []
    for i in range(handSize):
        hand.append(getCard('Card %d: ' % (i+1), age, players))
    return hand

def updateHandState(state, handSize):
    for i in range(state.numPlayers):
        state.players[i].hand = []
    state.players[0].hand = getHand(handSize, state.age, state.numPlayers)
    for i in range(state.numPlayers):
        print(state.players[i].hand)

class FakeBot:
    def __init__(self):
        self.name = 'Fake bot'
        pass

    def getPayOption(self, state):
        payBank = getNumber('How much did %s pay to the bank? ' % (self.wonder.name))
        payLeft = getNumber('How much did %s pay to %s? ' % (self.wonder.name, state.players[1].wonder.name))
        payRight = getNumber('How much did %s pay to %s? ' % (self.wonder.name, state.players[-1].wonder.name))
        return PayOption(payBank = payBank, payLeft = payLeft, payRight = payRight)

    def getMove(self, state):
        playedFaceUp = getBool('Did %s play a card face up? ' % self.wonder.name)
        if playedFaceUp:
            card = getCard('Which card did %s play? ' % self.wonder.name, state.age, state.numPlayers)
            payOption = self.getPayOption(state)
            return Move(card, payOption = payOption)


def playGame():
    players = getNumber('Number of players: ')
    bot = DNNReferenceBot(players)
    bot.testingMode = True
    bot.PRINT = PRINT
    bots = [bot]
    for i in range(players-1):
        bots.append(FakeBot())
    state = State(bots, doShuffle = False)
    state.players[0].wonder = getWonder('My wonder: ')
    for i in range(1, players):
        state.players[i].wonder = getWonder('Wonder of player %d seats to the left: ' % i)
    for i in range(players):
        state.players[i].bot.wonder = state.players[i].wonder
    for age in range(1, 4):
        state.initAge(age)
        for pick in range(1, 7):
            print('Age %d Pick %d' % (age, pick))
            updateHandState(state, 8-pick)
            state.print()
            moves = []
            for j in range(players):
                player = state.players[j]
                inputState = state.getStateFromPerspective(j)
                moves.append(player.bot.getMove(inputState))
            state = state.performMoves(moves)
            print('\n')
        state.resolveWar()
    state = state.endGame()
    state.print()

playGame()

