import copy
import random
import time
from ortools.graph import pywrapgraph
from card_database import ALL_WONDERS, getCards, PURPLE_CARDS, MANUFACTURED_RESOURCES, NONMANUFACTURED_RESOURCES
from card import Color, ProductionEffect, Resource, RESOURCES, ScoreEffect, GoldEffect, Constant, CardCounter, TradingEffect, ScienceEffect, Science, MilitaryEffect, DefeatCounter, PayOption, WonderCounter, Wonder, Card
from random_bot import RandomBot
from science_bot import ScienceBot
from typing import Optional, List, Set

PRINT = True
PRINT_VERBOSE = False


class State:

    def __init__(self, bots):
        bots = copy.copy(bots)
        random.shuffle(bots)
        self.numPlayers = len(bots)
        wonders = random.sample(ALL_WONDERS, self.numPlayers)
        self.players = [Player(wonder, bot) for wonder, bot in zip(wonders, bots)]
        for i in range(self.numPlayers):
            self.players[i].leftNeighbor = self.players[(i + 1) % self.numPlayers]
            self.players[i].rightNeighbor = self.players[(i - 1) % self.numPlayers]

    def initAge(self, age):
        self.age = age
        cards = getCards(age=self.age, players=len(self.players))
        if age == 3:
            cards += random.sample(PURPLE_CARDS, self.numPlayers + 2)
        random.shuffle(cards)
        for i in range(self.numPlayers):
            self.players[i].hand = cards[i * 7:(i + 1) * 7]

    def getStateFromPerspective(self, perspective):
        state = copy.copy(self)
        state.players = [self.players[perspective]]
        for i in range(perspective + 1, perspective + state.numPlayers):
            state.players.append(self.players[i % state.numPlayers])
        return state

    def getHiddenState(self, perspective):
        state = copy.copy(self)
        state.players = [self.players[perspective]]
        for i in range(perspective + 1, perspective + state.numPlayers):
            state.players.append(self.players[i % state.numPlayers].convertToHidden())
        return state

    def countCards(self, player, color):
        count = 0
        for card in self.players[player].boughtCards:
            if card.color == color:
                count += 1
        return count

    def evaluateCounter(self, counter, player):
        if isinstance(counter, Constant):
            return counter.value
        if isinstance(counter, CardCounter):
            count = 0
            if counter.countSelf:
                count += self.countCards(player, counter.color)
            if counter.countNeighbors:
                count += self.countCards(player - 1, counter.color)
                count += self.countCards((player + 1) % self.numPlayers, counter.color)
            return count * counter.multiplier
        if isinstance(counter, DefeatCounter):
            count = 0
            if counter.countSelf:
                count += len(self.players[player].militaryDefeats)
            if counter.countNeighbors:
                count += len(self.players[player - 1].militaryDefeats)
                count += len(self.players[(player + 1) % self.numPlayers].militaryDefeats)
            return count * counter.multiplier
        if isinstance(counter, WonderCounter):
            count = 0
            if counter.countSelf:
                count += self.players[player].numWonderStagesBuilt
            if counter.countNeighbors:
                count += self.players[player - 1].numWonderStagesBuilt
                count += self.players[(player + 1) % self.numPlayers].numWonderStagesBuilt
            return count * counter.multiplier
        return 0

    def getScienceScore(self, effects, ind=0, compasses=0, tablets=0, cogs=0):
        if ind == len(effects):
            return compasses * compasses + tablets * tablets + cogs * cogs + 7 * min(compasses, tablets, cogs)
        bestScore = 0
        if Science.COMPASS in effects[ind].symbols:
            bestScore = max(bestScore, self.getScienceScore(effects, ind + 1, compasses + 1, tablets, cogs))
        if Science.TABLET in effects[ind].symbols:
            bestScore = max(bestScore, self.getScienceScore(effects, ind + 1, compasses, tablets + 1, cogs))
        if Science.COG in effects[ind].symbols:
            bestScore = max(bestScore, self.getScienceScore(effects, ind + 1, compasses, tablets, cogs + 1))
        return bestScore

    def getScore(self, playerInd):
        player = self.players[playerInd]
        score = player.gold // 3
        scienceEffects = []
        for card in player.boughtCards:
            for effect in card.effects:
                if isinstance(effect, ScoreEffect):
                    score += self.evaluateCounter(counter=effect.counter, player=playerInd)
                elif isinstance(effect, ScienceEffect):
                    scienceEffects.append(effect)
        for i in range(player.numWonderStagesBuilt):
            for effect in player.wonder.stages[i].effects:
                if isinstance(effect, ScoreEffect):
                    score += self.evaluateCounter(counter=effect.counter, player=playerInd)
                elif isinstance(effect, ScienceEffect):
                    scienceEffects.append(effect)
        if PRINT_VERBOSE:
            print('Science score: %d' % self.getScienceScore(scienceEffects))
        score += self.getScienceScore(scienceEffects)
        if PRINT_VERBOSE:
            print('Military score: %d' % player.getMilitaryScore())
        score += player.getMilitaryScore()
        return score

    def applyEffect(self, effect, player):
        if isinstance(effect, GoldEffect):
            value = self.evaluateCounter(counter=effect.counter, player=player)
            if PRINT:
                print('%s received %d gold from the bank' % (self.players[player].name, value))
            self.players[player].gold += value
        elif isinstance(effect, TradingEffect):
            self.players[player].tradingEffects.append(effect)

    def performMoves(self, moves):
        state = copy.copy(self)
        oldHands = []
        for i in range(state.numPlayers):
            state.players[i].performMove(moves[i])
            oldHands.append(state.players[i].hand)
        for i in range(state.numPlayers):
            if moves[i].discard:
                continue
            if moves[i].buildWonder:
                for effect in state.players[i].wonder.stages[moves[i].wonderStageIndex].effects:
                    state.applyEffect(effect, i)
                continue
            for effect in moves[i].card.effects:
                state.applyEffect(effect, i)
        for i in range(state.numPlayers):
            if (state.age == 2):
                state.players[i].hand = oldHands[(i + 1) % len(oldHands)]
            else:
                state.players[i].hand = oldHands[i - 1]
        return state

    def print(self):
        for i in range(self.numPlayers):
            player = self.players[i]
            player.print(self.getScore(i))

    def endGame(self):
        state = copy.copy(self)
        for i in range(state.numPlayers):
            state.players[i].throwAwayHand()
        return state

    def getCardPayOptions(self, player, card):
        if card.chainFromNames.isdisjoint(self.players[player].boughtCardNames):
            return self.getPayOptionsForCost(player, card.cost)
        else:
            return {PayOption(isChained=True)}

    def getPayOptionsForCost(self, player, cost):
        payOptions = set()
        for payOption in self.getPayOptions(player, cost.resources):
            payOption.payBank += cost.gold
            payOptions.add(payOption)
        return payOptions

    def getPayOptions(self, player, resources):
        need = dict()
        for resource in RESOURCES:
            need[resource] = 0
        for resource in resources:
            need[resource] += 1
        production: List[List[ProductionEffect]] = [[], [], []]
        for i in range(-1, 2):
            p = self.players[(player + i) % self.numPlayers]
            effects = [p.wonder.effect]
            for card in p.boughtCards:
                if i != 0 and card.color != Color.BROWN and card.color != Color.GREY:
                    continue
                effects += card.effects
            for effect in effects:
                if isinstance(effect, ProductionEffect):
                    # effect.print()
                    production[i].append(effect.produces)

        payOptions = self.getPayOptionsSplit(player, need, production)
        if False:
            reducedPayOptions = self.getReducedPayOptions(player, need, production)
            # for option in reducedPayOptions:
            #    option.print()
            for payOption in payOptions:
                assert(payOption in reducedPayOptions)
            for payOption in reducedPayOptions:
                exists = False
                for l in range(payOption.payLeft + 1):
                    for r in range(payOption.payRight + 1):
                        if(PayOption(payOption.payBank, l, r) in payOptions):
                            exists = True
                assert(exists)
        return payOptions

    def costOfBuying(self, player, direction, resource):
        if direction == 0:
            return 0
        for effect in self.players[player].tradingEffects:
            if resource in effect.resources and ((direction == 1 and effect.leftNeighbor) or (direction == 2 and effect.rightNeighbor)):
                return 1
        return 2

    def getPayOptionsSplit(self, player, need, production):
        resources = [MANUFACTURED_RESOURCES, NONMANUFACTURED_RESOURCES]
        needSplit = [[], []]
        productionSplit = [[], []]
        for resource, amount in need.items():
            for i in range(2):
                if resource in resources[i]:
                    if amount > 0:
                        needSplit[i].append([resource, amount])
        payOptions = []
        for i in range(2):
            productionSplit[i] = [[], [], []]
            for j in range(3):
                for p in production[j]:
                    if p[0] in resources[i]:
                        productionSplit[i][j].append(p)
            payOptions.append(self.getPayOptionsOptimized(player, needSplit[i], productionSplit[i]))
        combinedPayOptions = set()
        for p1 in payOptions[0]:
            for p2 in payOptions[1]:
                combinedPayOptions.add(p1 + p2)
        return combinedPayOptions

    def getPayOptionsOptimized(self, player, need, production, limit1=100, limit2=100):
        minCostFlow = pywrapgraph.SimpleMinCostFlow()
        totNeed = 0
        numNeed = len(need)
        for i in range(numNeed):
            [resource, amount] = need[i]
            totNeed += amount
            minCostFlow.SetNodeSupply(i, -amount)
        minCostFlow.SetNodeSupply(numNeed, 100)
        minCostFlow.SetNodeSupply(numNeed + 1, limit1)
        minCostFlow.SetNodeSupply(numNeed + 2, limit2)
        nodeCount = numNeed + 3
        for i in range(3):
            sourceNode = numNeed + i
            for j in range(len(production[i])):
                costOfUse = self.costOfBuying(player, i, production[i][j][0])
                minCostFlow.AddArcWithCapacityAndUnitCost(sourceNode, nodeCount, 1, costOfUse)
                for k in range(numNeed):
                    [resource, amount] = need[k]
                    if resource in production[i][j]:
                        minCostFlow.AddArcWithCapacityAndUnitCost(nodeCount, k, 1, 0)
                nodeCount += 1
        minCostFlow.SolveMaxFlowWithMinCost()
        boughtFromPlayer = [0, 0, 0]
        payToPlayer = [0, 0, 0]
        for i in range(minCostFlow.NumArcs()):
            fromNode = minCostFlow.Tail(i)
            if fromNode >= numNeed and fromNode < numNeed + 3:
                boughtFromPlayer[fromNode - numNeed] += minCostFlow.Flow(i)
                payToPlayer[fromNode - numNeed] += minCostFlow.Flow(i) * minCostFlow.UnitCost(i)
        if sum(boughtFromPlayer) < totNeed:
            return set()
        payOptions = {PayOption(payToPlayer[0], payToPlayer[1], payToPlayer[2])}
        if boughtFromPlayer[1] > 0:
            payOptions.update(self.getPayOptionsOptimized(player, need, production, boughtFromPlayer[1] - 1, limit2))
        if boughtFromPlayer[2] > 0:
            payOptions.update(self.getPayOptionsOptimized(player, need, production, limit1, boughtFromPlayer[2] - 1))
        return payOptions

    def getReducedPayOptions(self, player, need, production):
        neededResource = None
        for resource, amount in need.items():
            if amount > 0:
                neededResource = resource
        if neededResource is None:
            return {PayOption()}
        payOptions = set()
        stopSearch = False
        for i in range(0, 3):
            for j in range(len(production[i])):
                if stopSearch:
                    break
                if neededResource in production[i][j]:
                    newNeed = copy.copy(need)
                    newNeed[neededResource] -= 1
                    newProduction = copy.deepcopy(production)
                    newProduction[i].pop(j)
                    costOfUse = self.costOfBuying(player, i, neededResource)
                    options = self.getReducedPayOptions(player, newNeed, newProduction)
                    for payOption in options:
                        if i == 1:
                            payOption.payLeft += costOfUse
                        elif i == 2:
                            payOption.payRight += costOfUse
                        payOptions.add(payOption)
                    if i == 0 and len(production[i][j]) == 1:
                        stopSearch = True
                    if len(production[i][j]) == 1:
                        break
        return payOptions

    def resolveWar(self):
        if self.age == 1:
            scoreForVictory = 1
        elif self.age == 2:
            scoreForVictory = 3
        elif self.age == 3:
            scoreForVictory = 5
        for i in range(0, self.numPlayers):
            j = (i + 1) % self.numPlayers
            shieldsI = self.players[i].getNumShields()
            shieldsJ = self.players[j].getNumShields()
            if shieldsI > shieldsJ:
                self.players[i].militaryVictories.append(scoreForVictory)
                self.players[j].militaryDefeats.append(1)
                if PRINT:
                    print('%s defeated %s %d-%d' % (self.players[i].name, self.players[j].name, shieldsI, shieldsJ))
            elif shieldsJ > shieldsI:
                self.players[j].militaryVictories.append(scoreForVictory)
                self.players[i].militaryDefeats.append(1)
                if PRINT:
                    print('%s defeated %s %d-%d' % (self.players[j].name, self.players[i].name, shieldsJ, shieldsI))
        if PRINT:
            print('\n')


class Player:

    def __init__(self, wonder: Wonder, bot):
        self.wonder = wonder
        self.name = self.wonder.name + ' (' + bot.name + ')'
        self.bot = bot
        self.boughtCards: List[Card] = []
        self.boughtCardNames: Set[str] = set()
        self.gold: int = 3
        self.tradingEffects: List[TradingEffect] = []
        self.militaryVictories: List[int] = []
        self.militaryDefeats: List[int] = []
        self.tensors = []
        self.bestMoveScores = []
        self.numWonderStagesBuilt = 0
        self.stateTensors = []
        self.allHandTensors = []
        self.leftNeighbor: Player = None
        self.rightNeighbor: Player = None

    def print(self, score):
        print(self.name)
        print('Score: %d' % score)
        print('Gold: %d' % self.gold)
        print('Bought cards')
        for card in self.boughtCards:
            card.print()
        if len(self.hand) > 0:
            print('\nHand')
            for card in self.hand:
                card.print()
        print('\n')

    def convertToHidden(self):
        player = copy.copy(self)
        player.hand = []
        return player

    def undoMove(self, move):
        if move.buildWonder:
            self.gold += move.payOption.totalCost()
            self.leftNeighbor.gold -= move.payOption.payLeft
            self.rightNeighbor.gold -= move.payOption.payRight
            self.numWonderStagesBuilt -= 1
        elif move.discard:
            self.gold -= 3
        else:
            self.gold += move.payOption.totalCost()
            self.leftNeighbor.gold -= move.payOption.payLeft
            self.rightNeighbor.gold -= move.payOption.payRight
            self.boughtCards.pop()
            self.boughtCardNames.remove(move.card.name)

    def performMove(self, move, removeCardFromHand=True):
        if PRINT and removeCardFromHand:
            self.printMove(move)
        if move.buildWonder:
            self.gold -= move.payOption.totalCost()
            self.leftNeighbor.gold += move.payOption.payLeft
            self.rightNeighbor.gold += move.payOption.payRight
            self.numWonderStagesBuilt += 1
        elif move.discard:
            self.gold += 3
        else:
            self.gold -= move.payOption.totalCost()
            self.leftNeighbor.gold += move.payOption.payLeft
            self.rightNeighbor.gold += move.payOption.payRight
            self.boughtCards.append(move.card)
            self.boughtCardNames.add(move.card.name)
        if removeCardFromHand:
            self.removeCardFromHand(move.card)

    def printMove(self, move):
        if (move.discard):
            print('%s discarded %s for 3 gold' % (self.name, move.card.name))
        elif move.buildWonder:
            if move.wonderStageIndex == 0:
                order = '1st'
            elif move.wonderStageIndex == 1:
                order = '2nd'
            elif move.wonderStageIndex == 2:
                order = '3rd'
            elif move.wonderStageIndex == 3:
                order = '4th'
            print('%s built their %s wonder stage %s' % (self.name, order, move.payOption.toString()))
        else:
            payingLeftString = (str(' paying %d gold to the left' % move.payOption.payLeft) if move.payOption.payLeft > 0 else '')
            payingRightString = (str(' paying %d gold to the right' % move.payOption.payRight) if move.payOption.payRight > 0 else '')
            payingString = (str('%s and%s' % (payingLeftString, payingRightString)) if move.payOption.payLeft >
                            0 and move.payOption.payRight > 0 else str('%s%s' % (payingLeftString, payingRightString)))
            chainingString = ' using chaining' if move.payOption.isChained else ''
            print('%s bought %s%s%s' % (self.name, move.card.name, payingString, chainingString))

    def removeCardFromHand(self, card):
        for i in range(len(self.hand)):
            if (self.hand[i].name == card.name):
                self.hand.pop(i)
                return

    def throwAwayHand(self):
        self.hand = []

    def getPlayableCards(self):
        return list(filter(lambda card: (card.name not in self.boughtCardNames), self.hand))

    def getNumShields(self):
        shields = 0
        for card in self.boughtCards:
            for effect in card.effects:
                if isinstance(effect, MilitaryEffect):
                    shields += effect.shields
        return shields

    def getMilitaryScore(self):
        score = 0
        for victoryScore in self.militaryVictories:
            score += victoryScore
        for defeatScore in self.militaryDefeats:
            score -= defeatScore
        return score


def playGame(bots):
    playGames(bots, 1)


def playGames(bots, numGames):
    startTime = time.time()
    players = len(bots)
    for i in range(players):
        bots[i].PRINT = PRINT
    # random.seed(1)
    allStates = []
    for gameInd in range(numGames):
        allStates.append(State(bots))
    for age in range(1, 4):
        for gameInd in range(numGames):
            allStates[gameInd].initAge(age)
        for pick in range(1, 7):
            if PRINT:
                print('Age %d Pick %d' % (age, pick))
            inputStates = []
            botInd = []
            for bot in bots:
                inputStates.append([])
            for gameInd in range(numGames):
                botInd.append([])
                for i in range(players):
                    state = allStates[gameInd]
                    player = state.players[i]
                    for j in range(len(bots)):
                        if bots[j] == player.bot and len(inputStates[j]) <= gameInd:
                            newInputState = state.getStateFromPerspective(i)
                            inputStates[j].append(newInputState)
                            botInd[gameInd].append(j)
                            break
            chosenMoves = []
            for j in range(len(bots)):
                chosenMoves.append(bots[j].getMoves(inputStates[j]))
            for gameInd in range(numGames):
                moves = []
                state = allStates[gameInd]
                if PRINT:
                    state.print()
                for i in range(players):
                    player = state.players[i]
                    moves.append(chosenMoves[botInd[gameInd][i]][gameInd])
                allStates[gameInd] = state.performMoves(moves)
                if PRINT:
                    print('\n')
        for gameInd in range(numGames):
            allStates[gameInd].resolveWar()
    for bot in bots:
        bot.scores = []
    endTime = time.time()
    print("Games took %.3f seconds" % (endTime - startTime))
    startTime = time.time()
    for gameInd in range(numGames):
        state = allStates[gameInd].endGame()
        if PRINT:
            state.print()
        for i in range(players):
            state.players[i].bot.scores.append(state.getScore(i))
        for i in range(players):
            state.players[i].bot.train(state.getStateFromPerspective(i))
    endTime = time.time()
    print("Updating bots took %.3f seconds" % (endTime - startTime))
    # for i in range(players):
    #    print('%s: %d' % (bots[i].name, bots[i].score))
