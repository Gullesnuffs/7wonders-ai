import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from card import Card, Move, Color
from card_database import ALL_CARDS, getCardIndex, ALL_WONDERS, getAllCardsWithMultiplicities, getNumCardsWithMultiplicities, getCards, PURPLE_CARDS, DEFAULT
from control import State
from typing import List, Union, Optional
from control import Player
import math
from pytorch_bot import TorchBot
import random
import copy
from trainer_rnn import TrainerRNN
import os
from datetime import datetime
import subprocess
from nash import Nash, Bonus

class MoveScore:
    def __init__(self, move: Move, baseScore: float):
        self.move = move
        self.baseScore = baseScore
        self.numRollouts = 0
        self.totalRolloutScore = 0

class RolloutBot:
    def __init__(self, numPlayers: int, checkpoint_path: str, name: str, writeToTensorboard = True):
        self.numPlayers = numPlayers
        self.allCardsWithMultiplicities = getAllCardsWithMultiplicities(numPlayers)
        self.numCardsWithMultiplicities = getNumCardsWithMultiplicities(numPlayers)
        self.PRINT = False
        self.testingMode = False
        self.name = name
        self.baseBot = TorchBot(numPlayers, checkpoint_path, name + ' : TorchBot', False)
        self.rolloutBot = TorchBot(numPlayers, checkpoint_path, name + ' : RolloutBot', False)
        self.baseBot.testingMode = True
        self.rolloutBot.testingMode = True
        self.rolloutBot.alwaysPickBestMoveInTestingMode = False
        self.rolloutCount = 1000
        self.hands = []
        self.debugPrint = False
        self.rating = 1000

    def onGameStart(self, numGames: int) -> None:
        self.baseBot.onGameStart(numGames)
        self.hands = []

    def getBonus(self):
        return Bonus()

    def onGameFinished(self, states: List[State]) -> None:
        self.baseBot.onGameFinished(states)

    def onRatingsAssigned(self) -> None:
        pass

    def observe(self, states: List[State]):
        pass

    def getStateValue(self, state):
        scores = [state.getScore(j) for j in range(state.numPlayers)]
        actualScore = 0
        wonGame = True
        for j in range(1, state.numPlayers):
            scoreDiff = scores[0] - scores[j]
            if scoreDiff != 0:
                scoreDiff += 5 if scoreDiff > 0 else -5
            value = 0.5 + 0.5 * np.tanh(scoreDiff * 0.05)
            if scores[j] >= scores[0]:
                wonGame = False
            actualScore += value / (state.numPlayers - 1)
        winValue = 0.4
        actualScore *= (1 - winValue)
        if wonGame:
            actualScore += winValue
        return actualScore

    def doRollout(self, state, chosenMoveScore):
        self.rolloutBot.gamesPlayed = 10000
        hands = [[], [], []]
        handsInTurn = []
        for age in range(3):
            for j in range(state.numPlayers):
                hands[age].append([])
            for moveInAge in range(6):
                handsInTurn.append([])

        for age in [2, 1, 0]:
            for moveInAge in [5, 4, 3, 2, 1, 0]:
                moveNum = age*6+moveInAge
                if moveNum >= len(self.hands):
                    continue
                for j in range(state.numPlayers):
                    if moveNum < len(state.playerMoveHistory[j]):
                        move = state.playerMoveHistory[j][moveNum]
                        if move.card != DEFAULT:
                            hands[age][j].append(move.card)

                if self.debugPrint:
                    print('Seen hand at age %d, pick %d:' % (age+1, moveInAge+1))
                    for card in self.hands[moveNum]:
                        print(card.name)
                for card in self.hands[moveNum]:
                    while self.hands[moveNum].count(card) > hands[age][0].count(card):
                        hands[age][0].append(card)
                #hands[age][0] = copy.copy(self.hands[moveNum])
                if moveInAge > 0:
                    oldHands = copy.copy(hands[age])
                    for j in range(state.numPlayers):
                        if (age == 1):
                            hands[age][j] = oldHands[j - 1]
                        else:
                            hands[age][j] = oldHands[(j + 1) % len(oldHands)]

            remainingCards = []
            cards = getCards(age+1, state.numPlayers)
            if age == 2:
                cards += PURPLE_CARDS
            for card in ALL_CARDS:
                remainingCards.append((card, cards.count(card)))
            seenPurpleCards = 0
            if self.debugPrint:
                print('Cards in hand:')
            for j in range(state.numPlayers):
                for cardInHand in hands[age][j]:
                    if self.debugPrint:
                        print(cardInHand.name)
                    if (cardInHand.color == Color.PURPLE):
                        seenPurpleCards += 1
                    for k in range(len(remainingCards)):
                        (card, numCardInstances) = remainingCards[k]
                        if card == cardInHand:
                            remainingCards[k] = (card, numCardInstances-1)
            numUnseenPurpleCards = len(PURPLE_CARDS) - seenPurpleCards
            numNewPurpleCards = state.numPlayers + 2 - seenPurpleCards
            for k in range(len(remainingCards)):
                (card, numCardInstances) = remainingCards[k]
                if card.color == Color.PURPLE and numCardInstances > 0:
                    if random.randrange(numUnseenPurpleCards) < numNewPurpleCards:
                        numNewPurpleCards -= 1
                    else:
                        remainingCards[k] = (card, numCardInstances-1)
                    numUnseenPurpleCards -= 1
            unknownCards = []
            for (card, numCardInstances) in remainingCards:
                if numCardInstances < 0:
                    print('Error: %d remaining instances of the card %s' % (numCardInstances, card.name))
                    assert(numCardInstances >= 0)
                for i in range(numCardInstances):
                    unknownCards.append(card)
            random.shuffle(unknownCards)
            if self.debugPrint:
                print('Unknown cards in age %d:' % (age+1))
                for card in unknownCards:
                    print(card.name)
                print('\n')

            # Fill out hands about which we have incomplete information using cards that haven't been seen yet.
            for j in range(state.numPlayers):
                if self.debugPrint:
                    print('Cards known to be in player %d\'s hand at the beginning of age %d:' % (j, age+1))
                    for card in hands[age][j]:
                        print(card.name)
                while len(hands[age][j]) < 7:
                    hands[age][j].append(unknownCards.pop())
                if self.debugPrint:
                    print('Cards assumed to be in player %d\'s hand at the beginning of age %d:' % (j, age+1))
                    for card in hands[age][j]:
                        print(card.name)
                    print('\n')
            assert(len(unknownCards) == 0)

            for moveInAge in range(6):
                moveNum = age*6+moveInAge
                if moveNum >= len(self.hands):
                    continue
                handsInTurn[moveNum] = []
                for j in range(state.numPlayers):
                    handsInTurn[moveNum].append(copy.copy(hands[age][j]))
                if self.debugPrint:
                    print('Hands in age %d, pick %d' % (age+1, moveInAge+1))
                    for j in range(state.numPlayers):
                        for card in handsInTurn[moveNum][j]:
                            print(card.name)
                        print('\n')
                    print('\n')
                for j in range(state.numPlayers):
                    assert(len(handsInTurn[moveNum][j]) == 7-moveInAge)
                for j in range(state.numPlayers):
                    if moveNum < len(state.playerMoveHistory[j]):
                        move = state.playerMoveHistory[j][moveNum]
                        if move.card == DEFAULT:
                            # The unseen cards are always at the end of the list
                            hands[age][j].pop()
                        else:
                            for k in range(len(hands[age][j])):
                                if (hands[age][j][k] == move.card):
                                    del hands[age][j][k]
                                    break
                oldHands = copy.copy(hands[age])
                for j in range(state.numPlayers):
                    if (age == 1):
                        hands[age][j] = oldHands[(j + 1) % len(oldHands)]
                    else:
                        hands[age][j] = oldHands[j - 1]

        #print('Start rollout')
        self.rolloutBot.onGameStart(state.numPlayers)
        currentState = state
        wonders = [state.players[i].wonder for i in range(state.numPlayers)]
        state = State(state.playerNames, wonders)
        for i in range(state.numPlayers):
            bonus = self.rolloutBot.getBonus()
            state.players[i].bonus = bonus
            state.players[i].scienceBonus = bonus.scienceBonus
            state.players[i].militaryBonus = bonus.militaryBonus

        for age in range(1, 4):
            state.initAge(age)

            for pick in range(1, 7):
                moveNum = (age-1)*6+(pick-1)
                if moveNum < len(self.hands):
                    for i in range(state.numPlayers):
                        state.players[i].hand = handsInTurn[moveNum][i]
                if self.debugPrint:
                    print('Rollout Age %d Pick %d' % (age, pick))
                    state.print()
                    print('\n')
                for i in range(state.numPlayers):
                    assert(len(state.players[i].hand) == 8-pick)

                inputStates = [state.getStateFromPerspective(playerIndex) for playerIndex in range(state.numPlayers)]
                if moveNum < len(currentState.playerMoveHistory[0]):
                    moves = self.rolloutBot.getMoves(inputStates)
                    moves = [currentState.playerMoveHistory[i][moveNum] for i in range(state.numPlayers)]
                else:
                    moves = self.rolloutBot.getMoves(inputStates)
                if moveNum == len(currentState.playerMoveHistory[0]):
                    moves[0] = chosenMoveScore.move
                state = state.performMoves(moves, doPrint = self.debugPrint)
            state.resolveWar(doPrint = False)
        self.rolloutBot.onGameFinished([state])
        chosenMoveScore.totalRolloutScore += self.getStateValue(state)
        chosenMoveScore.numRollouts += 1

    def getMove(self, state: State):
        #print('getMove from Rollout bot')
        self.hands.append(copy.copy(state.players[0].hand))
        baseMoveScores = self.baseBot.getMoveScores([state])[0]
        moveScores = [MoveScore(moveScore.move, moveScore.priority) for moveScore in baseMoveScores]
        remainingCards = {card: numCardInstances for (card, numCardInstances) in self.allCardsWithMultiplicities}
        for i in range(state.numPlayers):
            for card in state.players[i].boughtCards:
                remainingCards[card] -= 1
        baseScoreRolloutValue = 30
        i = 0
        while True:
            maxBaseScore = moveScores[0].baseScore
            maxRolloutScore = 0
            maxNumRollouts = 0
            for moveScore in moveScores:
                maxRolloutScore = max(maxRolloutScore, moveScore.totalRolloutScore / (moveScore.numRollouts + 1.0))
                maxNumRollouts = max(maxNumRollouts, moveScore.numRollouts)
            baseScoreBias = maxBaseScore - maxRolloutScore

            bestMoveScore = None
            for moveScore in moveScores:
                valueTerm = (moveScore.totalRolloutScore + baseScoreBias * moveScore.numRollouts + moveScore.baseScore * baseScoreRolloutValue) / (moveScore.numRollouts + baseScoreRolloutValue)
                explorationTerm = 0.3 * math.sqrt(math.log(i+2.0) / (moveScore.numRollouts+2.0))
                score = valueTerm + explorationTerm
                if bestMoveScore is None or score > bestScore:
                    bestMoveScore = moveScore
                    bestScore = score
            i += 1
            self.doRollout(state, bestMoveScore)
            if (i > self.rolloutCount and bestMoveScore.numRollouts > max(self.rolloutCount/3, maxNumRollouts)) or bestMoveScore.numRollouts > self.rolloutCount:
                break
            if i%100 == 0:
                for moveScore in moveScores:
                    print('Rollouts: %d\tBase score: %.4f\tAverage rollout score: %.4f\tMove: %s' % (moveScore.numRollouts, moveScore.baseScore, moveScore.totalRolloutScore / (moveScore.numRollouts + 1e-9), moveScore.move.toString()))
                print('\n')
        bestMoveScore = moveScores[0]
        for moveScore in moveScores:
            print('Rollouts: %d\tBase score: %.4f\tAverage rollout score: %.4f\tMove: %s' % (moveScore.numRollouts, moveScore.baseScore, moveScore.totalRolloutScore / (moveScore.numRollouts + 1e-9), moveScore.move.toString()))
            if moveScore.numRollouts > bestMoveScore.numRollouts:
                bestMoveScore = moveScore
        print('\n')
        return bestMoveScore.move

    def getMoves(self, states: List[State]):
        return [self.getMove(state) for state in states]
