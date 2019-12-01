import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from card import Card, Move
from card_database import ALL_CARDS, getCardIndex, ALL_WONDERS, getAllCardsWithMultiplicities, getNumCardsWithMultiplicities
from control import State
from typing import List, Union, Optional
from control import Player
import math
import random
from trainer_rnn import TrainerRNN
import os
from datetime import datetime
import subprocess

HIDDEN_STATE_SIZE = 512
ACTION_STATE_SIZE = len(ALL_CARDS) + 2*14 + 3
CHECKPOINT_PATH = 'pytorchbotreference/checkpoint.pt'


class TensorBoardWrapper:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.writer = None
        self.scalars = []

    def add_scalar(self, message, loss, step):
        if self.writer is not None:
            self.writer.add_scalar(message, loss, step)
        else:
            self.scalars.append((message, loss, step))

    def add_embedding(self, *args, **kwargs):
        self.writer.add_embedding(*args, **kwargs)

    def init(self):
        if self.writer is not None:
            return

        from tensorboardX import SummaryWriter
        self.writer = SummaryWriter(log_dir=self.log_dir)
        for s in self.scalars:
            self.writer.add_scalar(*s)
        self.scalars = None

    def set_epoch(self, epoch):
        if epoch >= 2:
            self.init()


class MoveScore:
    def __init__(self, move: Move, score: torch.tensor):
        self.move = move
        self.score = score
        self.priority = float(score.detach())


class Net(nn.Module):
    def __init__(self, stateSize: int):
        super(Net, self).__init__()
        self.lstm = nn.LSTMCell(stateSize, HIDDEN_STATE_SIZE)

        self.plin1 = nn.Linear(stateSize, 256)

        self.lin1 = nn.Linear(HIDDEN_STATE_SIZE + 256, 512)
        self.lin2 = nn.Linear(512, 256)

        self.alin1 = nn.Linear(ACTION_STATE_SIZE, 64)
        self.alin2 = nn.Linear(320, 256)
        self.alin3 = nn.Linear(256, 64)
        self.alin4 = nn.Linear(64, 1)

    def forward(self, x: torch.tensor, lstmState: torch.tensor, perActionStates: torch.tensor, actionToStateMapping: torch.tensor) -> torch.tensor:
        # Unpack the lstm states
        lstmState1 = lstmState[:, 0:HIDDEN_STATE_SIZE]
        lstmState2 = lstmState[:, HIDDEN_STATE_SIZE:]
        lstmState1, lstmState2 = self.lstm(x, (lstmState1, lstmState2))

        y = F.relu(self.plin1(x))

        x = torch.cat([y, lstmState1], dim=1)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))

        perActionCompressedStates = torch.index_select(x, dim=0, index=actionToStateMapping)

        perActionStates = self.alin1(perActionStates)
        perActionStates = torch.cat([perActionStates, perActionCompressedStates], dim=1)

        perActionStates = F.relu(self.alin2(perActionStates))
        perActionStates = F.relu(self.alin3(perActionStates))
        perActionStates = self.alin4(perActionStates)

        return perActionStates, torch.cat([lstmState1, lstmState2], axis=1)


def concat_smart(arr: List[Union[list, float, int]]) -> torch.tensor:
    arr = [x if isinstance(x, list) or isinstance(
        x, np.ndarray) else [float(x)] for x in arr]
    return torch.as_tensor(np.concatenate(arr), dtype=torch.float32)


class TensorBuilder():
    def __init__(self, capacity):
        self.capacity = capacity
        self.reset()

    def reset(self):
        self.index = 0
        self.tensor = np.zeros(self.capacity, dtype=np.float32)

    def onehot(self, size: int, index: int):
        assert index >= 0 and index < size
        self.tensor[self.index + index] = 1
        self.index += size

    def append(self, value):
        self.tensor[self.index] = float(value)
        self.index += 1

    def append_iter(self, iterator):
        for v in iterator:
            self.tensor[self.index] = float(v)
            self.index += 1

    def view(self, size: int):
        self.index += size
        return self.tensor[self.index - size:self.index]

    def value(self):
        return torch.as_tensor(self.tensor[:self.index])


class TorchReferenceBot:
    def __init__(self, numPlayers: int):
        self.numPlayers = numPlayers
        self.allCardsWithMultiplicities = getAllCardsWithMultiplicities(numPlayers)
        self.numCardsWithMultiplicities = getNumCardsWithMultiplicities(numPlayers)
        stateSize = self.getStateTensorDimension(self.numPlayers)
        self.model = Net(stateSize)
        if os.path.exists(CHECKPOINT_PATH):
            self.model.load_state_dict(torch.load(CHECKPOINT_PATH))
        self.gamesPlayed = 0
        self.PRINT = False
        self.testingMode = False
        self.name = "TorchReferenceBot"
        self.device = torch.device("cpu")
        self.trainer = TrainerRNN(optimizer=torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=0.000001), device=self.device)
        self.loss_function = nn.MSELoss(reduction="mean")
        self.last_move_scores: Optional[torch.tensor] = None

    def onGameStart(self, numGames: int) -> None:
        self.hiddenStates = torch.zeros((numGames, 2*HIDDEN_STATE_SIZE), dtype=torch.float32)
        self.total_loss = torch.zeros(1, requires_grad=False, device=self.device)
        self.last_move_scores = None
        self.trainer.reset()

    def onGameFinished(self, states: List[State]) -> None:
        final_scores = np.zeros((len(states), 1), dtype=np.float32)
        for i in range(len(states)):
            scores = [states[i].getScore(j) for j in range(states[i].numPlayers)]
            actualScore = 0
            wonGame = True
            for j in range(1, states[i].numPlayers):
                scoreDiff = scores[0] - scores[j]
                if scoreDiff != 0:
                    scoreDiff += 5 if scoreDiff > 0 else -5
                value = 0.5 + 0.5 * np.tanh(scoreDiff * 0.05)
                if scores[j] >= scores[0]:
                    wonGame = False
                actualScore += value / (states[i].numPlayers - 1)
            winValue = 0.4
            actualScore *= (1 - winValue)
            if wonGame:
                actualScore += winValue

            final_scores[i] = actualScore

            self.gamesPlayed += 1

        if not self.testingMode:
            self.backprop(None, torch.as_tensor(final_scores))

        # Normally normalized by number of steps
        self.total_loss = self.total_loss / (7*3)
        if not self.testingMode:
            self.trainer.backprop(self.total_loss)
            torch.save(self.model.state_dict(), CHECKPOINT_PATH)

    def onRatingsAssigned(self) -> None:
        pass

    @staticmethod
    def getPlayerStateTensorDimension():
        return len(ALL_CARDS) + len(ALL_WONDERS) + 26

    def getStateTensorDimension(self, numPlayers: int):
        return numPlayers*TorchReferenceBot.getPlayerStateTensorDimension() + self.numCardsWithMultiplicities + 10

    @staticmethod
    def getPlayerStateTensor(state: State, playerIndex: int, builder: TensorBuilder) -> None:
        GOLD_DIMENSION = 15

        player = state.players[playerIndex]

        myMilitaryScore = player.getMilitaryScore()
        leftMilitaryScore = state.players[(
            playerIndex + 1) % len(state.players)].getMilitaryScore()
        rightMilitaryScore = state.players[(
            playerIndex - 1) % len(state.players)].getMilitaryScore()

        multihot_cards = builder.view(len(ALL_CARDS))
        for card in player.boughtCards:
            multihot_cards[card.cardId] = 1
        builder.onehot(len(ALL_WONDERS), ALL_WONDERS.index(player.wonder))
        builder.onehot(GOLD_DIMENSION, min(GOLD_DIMENSION - 1, player.gold))
        builder.append((player.gold - 5.0) / 5.0)
        builder.append(player.getNumShields() / 5.0)
        builder.append(player.getMilitaryScore())
        builder.append(myMilitaryScore > leftMilitaryScore)
        builder.append(myMilitaryScore > rightMilitaryScore)
        builder.append(myMilitaryScore / 8.0)
        builder.append((state.getScore(playerIndex) - 20.0) / 20.0)
        builder.append(player.numWonderStagesBuilt > 0)
        builder.append(player.numWonderStagesBuilt > 1)
        builder.append(player.numWonderStagesBuilt > 2)
        builder.append(player.numWonderStagesBuilt > 3)

    def getHandTensor(self, state: State, builder: TensorBuilder):
        for (card, numCardInstances) in self.allCardsWithMultiplicities:
            numInHand = state.players[0].hand.count(card)
            for i in range(numCardInstances):
                builder.append(1 if numInHand > i else 0)

    @staticmethod
    def getAgeAndPickTensor(state: State, builder: TensorBuilder):
        builder.onehot(3, state.age - 1)
        builder.onehot(7, len(state.players[0].hand) - 1)

    def getStateTensor(self, state: State):
        builder = TensorBuilder(self.getStateTensorDimension(state.numPlayers))
        TorchReferenceBot.getAgeAndPickTensor(state, builder)
        self.getHandTensor(state, builder)
        for player in range(state.numPlayers):
            TorchReferenceBot.getPlayerStateTensor(state, player, builder)
        return builder.value()

    @staticmethod
    def getMovesForPlayer(state: State):
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

    def observe(self, states: List[State]):
        # tensors = torch.as_tensor(np.stack([TorchReferenceBot.getStateTensor(state) for state in states]))
        # _, self.hiddenStates = self.model(tensors, self.hiddenStates)
        pass

    def backprop(self, chosen_move_scores: Optional[torch.tensor], expected_scores_for_best_move: torch.tensor):
        '''
        chosen_move_scores: Scores for the moves that were taken this turn
        expected_scores_for_best_move: Scores for the best moves that could be taken this turn
        '''
        if self.last_move_scores is not None:
            self.total_loss += self.loss_function(self.last_move_scores, expected_scores_for_best_move)
        self.last_move_scores = chosen_move_scores

    def getMoves(self, states: List[State]):
        allMoves = []
        allIndexRanges = []
        for stateInd, state in enumerate(states):
            moves = TorchReferenceBot.getMovesForPlayer(state)
            allMoves.append(moves)

        numTotalActions = sum(len(m) for m in allMoves)
        perActionStates = np.zeros((numTotalActions, ACTION_STATE_SIZE), dtype=np.float32)
        actionToStateMapping = np.zeros(numTotalActions, dtype=np.long)

        index = 0
        for stateIndex, moves in enumerate(allMoves):
            allIndexRanges.append(range(index, index + len(moves)))
            for move in moves:
                perActionStates[index, move.card.cardId] = 1
                offset = len(ALL_CARDS)
                for i in range(14):
                    if move.payOption.payLeft > i:
                        perActionStates[index, offset] = 1
                    offset += 1
                for i in range(14):
                    if move.payOption.payRight > i:
                        perActionStates[index, offset] = 1
                    offset += 1
                if move.buildWonder:
                    perActionStates[index, offset + 0] = 1
                elif move.discard:
                    perActionStates[index, offset + 1] = 1
                else:
                    perActionStates[index, offset + 2] = 1
                offset += 3
                actionToStateMapping[index] = stateIndex
                index += 1

        stateTensors = torch.as_tensor(np.stack([self.getStateTensor(state) for state in states]))
        allMoves = [move for moves in allMoves for move in moves]
        scores, self.hiddenStates = self.model(stateTensors, self.hiddenStates, torch.as_tensor(perActionStates), torch.as_tensor(actionToStateMapping))

        chosenMoves = []
        chosen_move_scores = []
        best_move_scores = []
        for indexRange in allIndexRanges:
            moveScores = [MoveScore(allMoves[i], scores[i]) for i in indexRange]
            moveScores.sort(key=lambda x: -x.priority)

            def adjust_move_priorities(moveScores: List[MoveScore]):
                n = self.gamesPlayed + 2
                targetScore = moveScores[0].priority + \
                    0.3 * math.sqrt(math.log(n) / n)
                if self.testingMode:
                    # Always pick best one
                    for i, moveScores in enumerate(moveScores):
                        moveScores.priority = 1 if i == 0 else 0
                    return
                for moveScore in moveScores:
                    moveScore.priority = (1.0 / (targetScore - moveScore.priority))**2
                    if moveScore.move.discard:
                        moveScore.priority *= 0.3
                    elif moveScore.move.buildWonder:
                        moveScore.priority *= 0.3

                # Normalize priorities
                tot = sum(move.priority for move in moveScores)
                for moveScore in moveScores:
                    moveScore.priority /= tot

            best_move_scores.append(moveScores[0].score)
            adjust_move_priorities(moveScores)

            if self.PRINT:
                print('Reference bot\'s scores:')
                for moveScore in moveScores:
                    print('%s: %.3f (probability = %.3f)' % (
                        moveScore.move.toString(), float(moveScore.score.detach()), moveScore.priority))
                print('')

            chosenMove = np.random.choice(
                moveScores, p=[moveScore.priority for moveScore in moveScores])

            chosen_move_scores.append(chosenMove.score)
            chosenMoves.append(chosenMove.move)

        if not self.testingMode:
            self.backprop(torch.stack(chosen_move_scores), torch.stack(best_move_scores))
        return chosenMoves