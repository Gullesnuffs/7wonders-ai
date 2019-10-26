import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from card import Card, Move
from card_database import ALL_CARDS, getCardIndex, ALL_WONDERS
from state import State
from typing import List
from control import Player
import math
import random


class MoveScore:

    def __init__(self, move, score, tensor):
        self.move = move
        self.score = score
        self.tensor = tensor


class Net(nn.Module):
    def __init__(self, stateSize):
        super(Net, self).__init__()
        self.lstm = nn.LSTMCell((stateSize,), 256)
        self.lin1 = nn.Linear(256, 128)
        self.lin2 = nn.Linear(128, 128)
        self.lin3 = nn.Linear(128, 64)

        #  historicalInputs = keras.Input(shape=(6, getStateTensorDimension(self.numPlayers),), name='historical')
        # y = layers.LSTM(256)(historicalInputs)

        # inputs = keras.Input(shape=(getMoveTensorDimension(self.numPlayers)), name='state')
        # x = layers.Dense(512, activation='relu', name='dense_1', kernel_regularizer=keras.regularizers.l2(l=0.01))(inputs)
        # x = layers.Dropout(0.3)(x)
        # #lstmOutput, stateH, stateC = layers.LSTM(64, return_state=True, name='lstm')(x)
        # #lstmState = [stateH, stateC]
        # x = layers.Concatenate()([y, x])
        # x = layers.Dense(512, activation='relu', name='dense_2', kernel_regularizer=keras.regularizers.l2(l=0.01))(x)
        # x = layers.Dropout(0.3)(x)
        # x = layers.Dense(512, activation='relu', name='dense_3', kernel_regularizer=keras.regularizers.l2(l=0.01))(x)
        # x = layers.Dropout(0.3)(x)
        # x = layers.Dense(512, activation='relu', name='dense_4', kernel_regularizer=keras.regularizers.l2(l=0.01))(x)
        # x = layers.Dropout(0.3)(x)
        # outputScore = layers.Dense(1, activation='sigmoid', name='outputScore')(x)

        # self.model = keras.Model(inputs=[inputs, historicalInputs], outputs=outputScore)
        # self.model.summary()
        # print(self.model.input)
        # #self.optimizer = tf.keras.optimizers.RMSprop()
        # self.optimizer = tf.keras.optimizers.Adam()

    def forward(self, x, lstmState1, lstmState2):
        lstmState1, lstmState2 = self.lstm(x, lstmState1, lstmState2)
        x = lstmState1
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        return x


class TorchBot:
    def __init__(self, numPlayers):
        self.numPlayers = numPlayers

        stateSize = getStateTensorDimension(self.numPlayers)
        self.model = Net(stateSize)
        self.gamesPlayed = 0

    def getStateTensorDimension(numPlayers):
        return 365

    def getMoves(self, states):
        pass

    def train(self, state):
        blah = 0
        blah = self.getPlayerStateTensor(None, None)
        blah.self = 0
        pass

    def getPlayerStateTensor(self, state: State, player):
        PER_CARD_DIMENSION = 17
        GOLD_DIMENSION = 15

        p = state.players[player]
        tensor = np.zeros((len(ALL_CARDS) + len(ALL_WONDERS) + GOLD_DIMENSION + 10))
        index = 0
        for i in range(len(ALL_CARDS)):
            card = ALL_CARDS[i]
            if card.name in p.boughtCardNames:
                tensor[index] = 1
            index += 1
        for i in range(len(ALL_WONDERS)):
            if p.wonder == ALL_WONDERS[i]:
                tensor[index] = 1
            index += 1
        tensor[index + min(GOLD_DIMENSION - 1, p.gold)] = 1
        index += GOLD_DIMENSION
        tensor[index] = (p.gold - 5.0) / 5.0
        index += 1
        tensor[index] = p.getNumShields() / 5.0
        index += 1
        myMilitaryScore = p.getMilitaryScore()
        leftMilitaryScore = state.players[(player + 1) % len(state.players)].getMilitaryScore()
        if myMilitaryScore > leftMilitaryScore:
            tensor[index] = 1
        elif myMilitaryScore < leftMilitaryScore:
            tensor[index] = -1
        rightMilitaryScore = state.players[player - 1].getMilitaryScore()
        index += 1
        if myMilitaryScore > rightMilitaryScore:
            tensor[index] = 1
        elif myMilitaryScore < rightMilitaryScore:
            tensor[index] = -1
        index += 1
        tensor[index] = p.getMilitaryScore() / 8.0
        index += 1
        tensor[index] = (state.getScore(player) - 20.0) / 20.0
        index += 1
        for i in range(4):
            if p.numWonderStagesBuilt > i:
                tensor[index] = 1
            index += 1
        return tensor

    def getHandTensor(self, state: State):
        tensor = np.zeros((len(ALL_CARDS)))
        index = 0
        for i in range(len(ALL_CARDS)):
            card = ALL_CARDS[i]
            if card in state.players[0].hand:
                tensor[index] = 1
            index += 1
        return tensor

    def getAgeAndPickTensor(self, state: State):
        tensor = np.zeros((9))
        tensor[state.age - 1] = 1
        tensor[10 - len(state.players[0].hand)] = 1
        return tensor

    def getStateTensor(self, state: State):
        tensor = np.concatenate((self.getAgeAndPickTensor(state), self.getHandTensor(state)))
        for player in range(state.numPlayers):
            tensor = np.concatenate((tensor, self.getPlayerStateTensor(state, player)))
        return tensor.astype(np.float32)

    class BotInstance:
        def __init__(self):
            self.hiddenState = []

    def getMovesForPlayer(self, state: State):
        moves = []
        player = state.players[0]
        for card in player.hand:
            moves.append(Move(card=card, discard=True))
            if player.numWonderStagesBuilt < len(player.wonder.stages):
                payOptions = state.getPayOptionsForCost(0, player.wonder.stages[player.numWonderStagesBuilt].cost)
                for payOption in payOptions:
                    if (payOption.totalCost() <= player.gold):
                        moves.append(Move(card=card, buildWonder=True, wonderStageIndex=player.numWonderStagesBuilt, payOption=payOption))
        for card in player.getPlayableCards():
            payOptions = state.getCardPayOptions(0, card)
            for payOption in payOptions:
                if (payOption.totalCost() <= player.gold):
                    moves.append(Move(card=card, payOption=payOption))
        return moves

    def getMoves(self, states: List[State]):
        allMoves = []
        allMoveScores = []
        allTensors = []
        for stateInd in range(len(states)):
            state = states[stateInd]
            # player = state.players[0]
            # player.stateTensors.append(self.getStateTensor(state))

            moves = self.getMovesForPlayer(state)
            moveScores = []
            tensors = []
            for move in moves:
                state.players[0].performMove(move, removeCardFromHand=False)
                tensor = self.getStateTensor(state)
                state.players[0].undoMove(move)
                tensors.append(tensor)
            allMoves.append(moves)
            allMoveScores.append(moveScores)
            allTensors.append(tensors)

        flatTensors = [tensor for tensorList in allTensors for tensor in tensorList]
        scores = self.model(torch.stack(flatTensors))
        scoreInd = 0
        chosenMoves = []
        for state, moves, moveScores, tensors in zip(states, allMoves, allMoveScores, allTensors):
            player = state.players[0]

            for move, tensor in zip(moves, tensors):
                score = scores[scoreInd][0]
                scoreInd += 1
                moveScores.append(MoveScore(move, score, tensor))

            moveScores.sort(key=lambda x: -x.score)
            bestMoveScore = moveScores[0].score
            totalPriority = 0
            n = self.gamesPlayed + 2
            targetScore = moveScores[0].score + 0.3 * math.sqrt(math.log(n) / n)
            if self.testingMode:
                targetScore = moveScores[0].score + 1e-6
            for moveScore in moveScores:
                moveScore.priority = 1.0 / (targetScore - moveScore.score)
                moveScore.priority *= moveScore.priority
                if moveScore.move.discard:
                    moveScore.priority *= 0.3
                elif moveScore.move.buildWonder:
                    moveScore.priority *= 0.3
                totalPriority += moveScore.priority
            targetPriority = random.random() * totalPriority
            if self.PRINT:
                print('DNN bot\'s scores:')
                for moveScore in moveScores:
                    print('%s: %.3f (probability = %.3f)' % (moveScore.move.toString(), moveScore.score, moveScore.priority / totalPriority))
                print('')
            for moveScore in moveScores:
                targetPriority -= moveScore.priority
                if targetPriority <= 0:
                    player.tensors.append(moveScore.tensor)
                    player.bestMoveScores.append(bestMoveScore)
                    chosenMoves.append(moveScore.move)
                    break
        return chosenMoves
