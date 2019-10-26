from enum import Enum
from termcolor import colored


class Color(Enum):
    BROWN = 1
    GREY = 2
    BLUE = 3
    YELLOW = 4
    RED = 5
    GREEN = 6
    PURPLE = 7


def colorToTermColor(color):
    if (color == Color.BROWN):
        return 'cyan'
    if (color == Color.GREY):
        return 'white'
    if (color == Color.BLUE):
        return 'blue'
    if (color == Color.YELLOW):
        return 'yellow'
    if (color == Color.RED):
        return 'red'
    if (color == Color.GREEN):
        return 'green'
    if (color == Color.PURPLE):
        return 'magenta'


class Resource(Enum):
    CLAY = 1
    ORE = 2
    STONE = 3
    WOOD = 4
    GLASS = 5
    CLOTH = 6
    PAPYRUS = 7


RESOURCES = [
    Resource.CLAY,
    Resource.ORE,
    Resource.STONE,
    Resource.WOOD,
    Resource.GLASS,
    Resource.CLOTH,
    Resource.PAPYRUS,
]


def resourceToString(resource):
    if (resource == Resource.CLAY):
        return 'c'
    if (resource == Resource.ORE):
        return 'o'
    if (resource == Resource.STONE):
        return 's'
    if (resource == Resource.WOOD):
        return 'w'
    if (resource == Resource.GLASS):
        return 'G'
    if (resource == Resource.CLOTH):
        return 'C'
    if (resource == Resource.PAPYRUS):
        return 'P'
    return 'UNKNOWN RESOURCE'


class Science(Enum):
    COMPASS = 1
    TABLET = 2
    COG = 3


class Cost:

    def __init__(self, gold=0, resources=[]):
        self.gold = gold
        self.resources = resources


class Counter:
    pass


class Constant(Counter):

    def __init__(self, value):
        self.value = value


class CardCounter(Counter):

    def __init__(self, color, countSelf=False, countNeighbors=False, multiplier=1):
        self.countSelf = countSelf
        self.countNeighbors = countNeighbors
        self.color = color
        self.multiplier = multiplier


class DefeatCounter(Counter):
    def __init__(self, countSelf=False, countNeighbors=False, multiplier=1):
        self.countSelf = countSelf
        self.countNeighbors = countNeighbors
        self.multiplier = multiplier


class WonderCounter(Counter):
    def __init__(self, countSelf=False, countNeighbors=False, multiplier=1):
        self.countSelf = countSelf
        self.countNeighbors = countNeighbors
        self.multiplier = multiplier


class ProductionEffect:

    def __init__(self, produces):
        self.produces = produces

    def print(self):
        producesString = ''
        for resource in self.produces:
            producesString += resourceToString(resource)
        print('Produces %s' % producesString)


class GoldEffect:

    def __init__(self, counter):
        self.counter = counter


class ScoreEffect:

    def __init__(self, counter):
        self.counter = counter


class ScienceEffect:

    def __init__(self, symbols):
        self.symbols = symbols


class MilitaryEffect:

    def __init__(self, shields):
        self.shields = shields


class TradingEffect:

    def __init__(self, resources, leftNeighbor=False, rightNeighbor=False):
        self.resources = resources
        self.leftNeighbor = leftNeighbor
        self.rightNeighbor = rightNeighbor


cardCount = 0


class Card:

    def __init__(self, name, color, cost=Cost(), effects=[], chainFrom=[]):
        self.name = name
        self.color = color
        self.cost = cost
        self.effects = effects
        self.chainFromNames = set()
        for card in chainFrom:
            self.chainFromNames.add(card.name)
        global cardCount
        self.cardId = cardCount
        cardCount += 1

    def print(self):
        print(self.toString())

    def toString(self):
        return colored(self.name, colorToTermColor(self.color))


class WonderStage:

    def __init__(self, cost, effects):
        self.cost = cost
        self.effects = effects


class Wonder:

    def __init__(self, name, effect, stages):
        self.name = name
        self.effect = effect
        self.stages = stages

    def print(self):
        print(self.name)


class PayOption:

    def __init__(self, payBank=0, payLeft=0, payRight=0, isChained=False):
        self.payBank = payBank
        self.payLeft = payLeft
        self.payRight = payRight
        self.isChained = isChained

    def totalCost(self):
        return self.payBank + self.payLeft + self.payRight

    def __eq__(self, other):
        return self.payBank == other.payBank and self.payLeft == other.payLeft and self.payRight == other.payRight

    def __hash__(self):
        return self.payBank * 10000 + self.payLeft * 100 + self.payRight

    def print(self):
        print('(%d %d %d)' % (self.payBank, self.payLeft, self.payRight))

    def __add__(self, o):
        return PayOption(self.payBank + o.payBank, self.payLeft + o.payLeft, self.payRight + o.payRight)

    def toString(self):
        payingLeftString = (str('%d to the left' % self.payLeft) if self.payLeft > 0 else '')
        payingRightString = (str('%d to the right' % self.payRight) if self.payRight > 0 else '')
        if self.payLeft > 0 and self.payRight > 0:
            payingString = str(' (%s, %s)' % (payingLeftString, payingRightString))
        elif self.payLeft > 0 or self.payRight > 0:
            payingString = str(' (%s%s)' % (payingLeftString, payingRightString))
        else:
            payingString = ''
        return ('for %d gold%s' % (self.totalCost(), payingString))


class Move:

    def __init__(self, card, payOption=PayOption(), discard=False, buildWonder=False, wonderStageIndex=0):
        self.card = card
        self.payOption = payOption
        self.discard = discard
        self.buildWonder = buildWonder
        self.wonderStageIndex = wonderStageIndex

    def toString(self):
        if self.discard:
            return 'Discard ' + self.card.name
        if self.buildWonder:
            if self.wonderStageIndex == 0:
                order = '1st'
            elif self.wonderStageIndex == 1:
                order = '2nd'
            elif self.wonderStageIndex == 2:
                order = '3rd'
            elif self.wonderStageIndex == 3:
                order = '4th'
            return ('Build %s wonder stage using %s %s' % (order, self.card.name, self.payOption.toString()))
        return ('%s %s' % (self.card.name, self.payOption.toString()))
