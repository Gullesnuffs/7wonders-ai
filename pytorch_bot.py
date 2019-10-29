import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from card import Card, Move
from card_database import ALL_CARDS, getCardIndex, ALL_WONDERS
from control import State
from typing import List, Union
from control import Player
import math
import random
from replay_memory import Transition, ReplayMemory

class MoveScore:
    def __init__(self, move: Move, score: float, tensor: torch.tensor):
        self.move = move
        self.score = score
        self.tensor = tensor
        self.priority = 0.0


class Net(nn.Module):
    def __init__(self, stateSize: int):
        super(Net, self).__init__()
        self.lstm = nn.LSTMCell((stateSize,), 256)
        self.lin1 = nn.Linear(256, 128)
        self.lin2 = nn.Linear(128, 128)
        self.lin3 = nn.Linear(128, 64)

    def forward(self, x: torch.tensor, lstmState1: torch.tensor, lstmState2: torch.tensor):
        lstmState1, lstmState2 = self.lstm(x, lstmState1, lstmState2)
        x = lstmState1
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        return x


def onehot(size: int, index: int) -> np.array:
    assert 0 <= index < size
    arr = np.zeros(size)
    arr[index] = 1
    return arr


def concat_smart(arr: List[Union[list, float, int]]) -> torch.tensor:
    arr = [x if isinstance(x, list) or isinstance(
        x, np.array) else [float(x)] for x in arr]
    return torch.cat(arr)


class TorchBot:
    def __init__(self, numPlayers: int):
        self.numPlayers = numPlayers

        stateSize = TorchBot.getStateTensorDimension(self.numPlayers)
        self.model = Net(stateSize)
        self.gamesPlayed = 0
        self.PRINT = False
        self.testingMode = False

    @staticmethod
    def getStateTensorDimension(numPlayers: int):
        return 365

    def train(self, state: State):
        replay_memory = ReplayMemory()

        batch_size = 128
        optimizer = torch.optim.Adam(self.model.parameters())
        # Standard Q learning
        while True:
            if len(replay_memory) > 1000:
                # Train
                replay_memory.sample(batch_size)
        pass

    def getPlayerStateTensor(self, state: State, playerIndex: int) -> torch.tensor:
        GOLD_DIMENSION = 15

        player = state.players[playerIndex]

        myMilitaryScore = player.getMilitaryScore()
        leftMilitaryScore = state.players[(
            playerIndex + 1) % len(state.players)].getMilitaryScore()
        rightMilitaryScore = state.players[(
            playerIndex - 1) % len(state.players)].getMilitaryScore()

        return concat_smart([
            [1 if card.name in player.boughtCardNames else 0 for card in ALL_CARDS],
            onehot(len(ALL_WONDERS), ALL_WONDERS.index(player.wonder)),
            onehot(GOLD_DIMENSION, min(GOLD_DIMENSION - 1, player.gold)),
            (player.gold - 5.0) / 5.0,
            player.getNumShields() / 5.0,
            player.getMilitaryScore(),
            myMilitaryScore > leftMilitaryScore,
            myMilitaryScore > rightMilitaryScore,
            myMilitaryScore / 8.0,
            (state.getScore(playerIndex) - 20.0) / 20.0,
            player.numWonderStagesBuilt > 0,
            player.numWonderStagesBuilt > 1,
            player.numWonderStagesBuilt > 2,
            player.numWonderStagesBuilt > 3,
        ])

    def getHandTensor(self, state: State):
        tensor = np.zeros((len(ALL_CARDS)))
        for i, card in enumerate(ALL_CARDS):
            tensor[i] = 1 if card in state.players[0].hand else 0
        return tensor

    def getAgeAndPickTensor(self, state: State):
        tensor = np.zeros((9))
        tensor[state.age - 1] = 1
        tensor[10 - len(state.players[0].hand)] = 1
        return tensor

    def getStateTensor(self, state: State):
        return torch.cat(
            [self.getAgeAndPickTensor(state), self.getHandTensor(state)] +
            [self.getPlayerStateTensor(state, player)
             for player in range(state.numPlayers)]
        )

    class BotInstance:
        def __init__(self):
            self.hiddenState = []

    def getMovesForPlayer(self, state: State):
        moves = []
        player = state.players[0]
        for card in player.hand:
            moves.append(Move(card=card, discard=True))
            if player.numWonderStagesBuilt < len(player.wonder.stages):
                payOptions = state.getPayOptionsForCost(
                    0, player.wonder.stages[player.numWonderStagesBuilt].cost)
                for payOption in payOptions:
                    if (payOption.totalCost() <= player.gold):
                        moves.append(Move(
                            card=card, buildWonder=True, wonderStageIndex=player.numWonderStagesBuilt, payOption=payOption))
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
            moves = self.getMovesForPlayer(state)
            moveScores: List[MoveScore] = []
            tensors = []
            for move in moves:
                state.players[0].performMove(move, removeCardFromHand=False)
                tensor = self.getStateTensor(state)
                state.players[0].undoMove(move)
                tensors.append(tensor)
            allMoves.append(moves)
            allMoveScores.append(moveScores)
            allTensors.append(tensors)

        flatTensors = [
            tensor for tensorList in allTensors for tensor in tensorList]
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

            def adjust_move_priorities(moveScores: List[MoveScore]):
                n = self.gamesPlayed + 2
                targetScore = moveScores[0].score + \
                    0.3 * math.sqrt(math.log(n) / n)
                if self.testingMode:
                    targetScore = moveScores[0].score + 1e-6
                for moveScore in moveScores:
                    moveScore.priority = 1.0 / (targetScore - moveScore.score)
                    moveScore.priority *= moveScore.priority
                    if moveScore.move.discard:
                        moveScore.priority *= 0.3
                    elif moveScore.move.buildWonder:
                        moveScore.priority *= 0.3

                # Normalize priorities
                tot = sum(move.priority for move in moveScores)
                for moveScore in moveScores:
                    moveScore.priority /= tot

            bestMoveScore = moveScores[0].score
            adjust_move_priorities(moveScores)

            if self.PRINT:
                print('DNN bot\'s scores:')
                for moveScore in moveScores:
                    print('%s: %.3f (probability = %.3f)' % (
                        moveScore.move.toString(), moveScore.score, moveScore.priority))
                print('')

            chosenMove = np.random.choice(
                moveScores, p=[moveScore.priority for moveScore in moveScores])
            player.tensors.append(chosenMove.tensor)
            chosenMoves.append(chosenMove.move)
            player.bestMoveScores.append(bestMoveScore)
        return chosenMoves
