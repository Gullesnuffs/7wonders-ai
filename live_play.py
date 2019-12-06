from dnn_reference_bot import DNNReferenceBot
from pytorch_bot import TorchBot, CardBonuses
from pytorch_bot2 import ParameterizedTorchBot
from card_database import getCards, PURPLE_CARDS, ALL_WONDERS, DEFAULT
from control import State
from card import PayOption, Move, Color

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
    considerationList = sorted(list(set(considerationList)), key = lambda item: item.name)
    while True:
        name = input(query)
        candidates = []
        for item in considerationList:
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

class FakeBot:
    def __init__(self):
        self.name = 'Fake bot'
        pass

    def getPayOption(self, state, card):
        options = state.getCardPayOptions(0, card)
        if len(options) == 0:
            print("There is no way to purchase this card")
            return None
        if len(options) == 1:
            opt = list(options)[0]
            print(f"Assuming {opt.payBank} to the bank, {opt.payLeft} to the left and {opt.payRight} to the right")
            return opt
        
        payBank = getNumber('How much did %s pay to the bank? ' % (state.players[0].name))
        payLeft = getNumber('How much did %s pay to %s? ' % (state.players[0].name, state.players[1].name))
        payRight = getNumber('How much did %s pay to %s? ' % (state.players[0].name, state.players[-1].name))
        opt = PayOption(payBank = payBank, payLeft = payLeft, payRight = payRight)
        f = [o for o in options if o.payBank == opt.payBank and o.payLeft == opt.payLeft and o.payRight == opt.payRight]
        if len(f) != 1:
            print("You cannot pay with that particular combination")
            return None
        return opt

    def getPayOptionSimple(self, state):
        payBank = getNumber('How much did %s pay to the bank? ' % (state.players[0].name))
        payLeft = getNumber('How much did %s pay to %s? ' % (state.players[0].name, state.players[1].name))
        payRight = getNumber('How much did %s pay to %s? ' % (state.players[0].name, state.players[-1].name))
        opt = PayOption(payBank = payBank, payLeft = payLeft, payRight = payRight)
        return opt

    def getMove(self, state):
        while True:
            playedFaceUp = getBool('Did %s play a card face up? ' % state.players[0].name)
            if playedFaceUp:
                card = getCard('Which card did %s play? ' % state.players[0].name, state.age, state.numPlayers)
                payOption = self.getPayOption(state, card)
                if payOption is None:
                    continue
                return Move(card, payOption = payOption)
            discarded = getBool('Did %s discard a card? ' % state.players[0].name)
            if discarded:
                return Move(DEFAULT, discard = True)
            buildWonder = getBool('Did %s build a wonder stage? ' % state.players[0].name)
            if buildWonder:
                payOption = self.getPayOptionSimple(state)
                return Move(DEFAULT, payOption = payOption, buildWonder = True, wonderStageIndex = state.players[0].numWonderStagesBuilt)


def playGame():
    players = getNumber('Number of players: ')
    torchBot = ParameterizedTorchBot(players, 'pytorchbot/parameterizedtorchbot.pt', 'ParameterizedTorchBot')
    bot = torchBot
    bot.testingMode = True
    bot.PRINT = PRINT
    bot.onGameStart(1)
    bots = [bot]
    for i in range(players-1):
        bots.append(FakeBot())
    wonders = []
    wonders.append(getWonder('My wonder: '))
    for i in range(1, players):
        wonders.append(getWonder('Wonder of player %d seats to the left: ' % i))
    state = State([wonder.name for wonder in wonders], wonders = wonders)
    for i in range(players):
        state.players[i].name = state.players[i].wonder.shortName
    state.players[0].scienceBonus = 0.0
    state.players[0].militaryBonus = 0.1
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
                moves.append(bots[j].getMove(inputState))
            state = state.performMoves(moves)
            print('\n')
        state.resolveWar()
    state = state.endGame()
    state.print()

playGame()

