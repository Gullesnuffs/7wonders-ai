import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from card import Card, Move
from card_database import ALL_CARDS, getCardIndex, ALL_WONDERS
from control import State
from typing import List, Union, Optional
from control import Player
import math
import random
from trainer_rnn import TrainerRNN


HIDDEN_STATE_SIZE = 256


class MoveScore:
    def __init__(self, move: Move, score: torch.tensor):
        self.move = move
        self.score = score
        self.priority = float(score.detach())


class Net(nn.Module):
    def __init__(self, stateSize: int):
        super(Net, self).__init__()
        self.lstm = nn.LSTMCell(stateSize, HIDDEN_STATE_SIZE)
        self.lin1 = nn.Linear(HIDDEN_STATE_SIZE, 128)
        self.lin2 = nn.Linear(128, 128)
        self.lin3 = nn.Linear(128, 64)
        self.lin4 = nn.Linear(64, 1)

    def forward(self, x: torch.tensor, lstmState: torch.tensor) -> torch.tensor:
        # Unpack the lstm states
        lstmState1 = lstmState[:, 0:HIDDEN_STATE_SIZE]
        lstmState2 = lstmState[:, HIDDEN_STATE_SIZE:]
        lstmState1, lstmState2 = self.lstm(x, (lstmState1, lstmState2))
        x = lstmState1
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        x = self.lin4(x)
        return x, torch.cat([lstmState1, lstmState2], axis=1)


def onehot(size: int, index: int) -> np.ndarray:
    assert 0 <= index < size
    arr = np.zeros(size)
    arr[index] = 1
    return arr


def concat_smart(arr: List[Union[list, float, int]]) -> torch.tensor:
    arr = [x if isinstance(x, list) or isinstance(
        x, np.ndarray) else [float(x)] for x in arr]
    return torch.as_tensor(np.concatenate(arr), dtype=torch.float32)


class TorchBot:
    def __init__(self, numPlayers: int):
        self.numPlayers = numPlayers
        stateSize = TorchBot.getStateTensorDimension(self.numPlayers)
        self.model = Net(stateSize)
        self.gamesPlayed = 0
        self.PRINT = False
        self.testingMode = False
        self.name = "TorchBot"
        self.device = torch.device("cpu")
        self.trainer = TrainerRNN(optimizer=torch.optim.Adam(self.model.parameters()), device=self.device)
        self.loss_function = nn.MSELoss(reduction="mean")
        self.last_move_scores: Optional[torch.tensor] = None

    def onGameStart(self, numGames: int) -> None:
        self.hiddenStates = torch.zeros((numGames, 2*HIDDEN_STATE_SIZE), dtype=torch.float32)
        self.total_loss = torch.zeros(1, requires_grad=False, device=self.device)
        self.last_move_scores = None
        self.trainer.reset()

    def onGameFinished(self, states: List[State]) -> None:
        # Normally normalized by number of steps... but it's the same for all games so whatever
        final_scores = np.zeros((len(states), 1), dtype=np.float32)
        for i in range(len(states)):
            final_scores[i] = states[i].getScore(0)
        self.backprop(None, torch.as_tensor(final_scores))
        self.trainer.backprop(self.total_loss)

    @staticmethod
    def getStateTensorDimension(numPlayers: int):
        return 368

    @staticmethod
    def getPlayerStateTensor(state: State, playerIndex: int) -> torch.tensor:
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

    @staticmethod
    def getHandTensor(state: State):
        tensor = np.zeros((len(ALL_CARDS)), dtype=np.float32)
        for i, card in enumerate(ALL_CARDS):
            tensor[i] = 1 if card in state.players[0].hand else 0
        return tensor

    @staticmethod
    def getAgeAndPickTensor(state: State):
        tensor = np.zeros((9), dtype=np.float32)
        tensor[state.age - 1] = 1
        tensor[10 - len(state.players[0].hand)] = 1
        return tensor

    @staticmethod
    def getStateTensor(state: State):
        return torch.as_tensor(np.concatenate(
            [TorchBot.getAgeAndPickTensor(state), TorchBot.getHandTensor(state)] +
            [TorchBot.getPlayerStateTensor(state, player)
             for player in range(state.numPlayers)]
        ))

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
        tensors = torch.as_tensor(np.stack([TorchBot.getStateTensor(state) for state in states]))
        _, self.hiddenStates = self.model(tensors, self.hiddenStates)

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
        allTensors = []
        allHiddenStates = []
        allIndexRanges = []
        for stateInd, state in enumerate(states):
            moves = TorchBot.getMovesForPlayer(state)

            allIndexRanges.append(range(len(allTensors), len(allTensors) + len(moves)))
            for move in moves:
                state.players[0].performMove(move, removeCardFromHand=False)
                tensor = TorchBot.getStateTensor(state)
                state.players[0].undoMove(move)
                allTensors.append(tensor)
                allMoves.append(move)
                allHiddenStates.append(self.hiddenStates[stateInd])

        scores, _ = self.model(torch.as_tensor(np.stack((allTensors))), torch.stack(allHiddenStates))
        chosenMoves = []
        chosen_move_scores = []
        best_move_scores = []
        for state, indexRange in zip(states, allIndexRanges):
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
                print('DNN bot\'s scores:')
                for moveScore in moveScores:
                    print('%s: %.3f (probability = %.3f)' % (
                        moveScore.move.toString(), float(moveScore.score.detach()), moveScore.priority))
                print('')

            chosenMove = np.random.choice(
                moveScores, p=[moveScore.priority for moveScore in moveScores])

            chosen_move_scores.append(chosenMove.score)
            chosenMoves.append(chosenMove.move)

        self.backprop(torch.stack(chosen_move_scores), torch.stack(best_move_scores))
        return chosenMoves
