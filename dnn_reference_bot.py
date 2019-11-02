from __future__ import absolute_import, division, print_function

import tensorflow as tf

# tf.enable_eager_execution()

import compress_pickle
import numpy as np
import random
import math
import time
import os.path
#from tensorflow import contrib
from tensorflow import keras
from tensorflow.keras import layers
from card import Card, Move
from card_database import ALL_CARDS, getCardIndex, ALL_WONDERS
from control import State

#tfe = contrib.eager


def getStateTensorDimension(numPlayers):
    return 365


def getMoveTensorDimension(numPlayers):
    return 450
    # return numPlayers * (len(ALL_CARDS)+len(ALL_WONDERS)+GOLD_DIMENSION) + len(ALL_CARDS)


class MoveScore:

    def __init__(self, move, score, tensor):
        self.move = move
        self.score = score
        self.tensor = tensor


PER_CARD_DIMENSION = 17
GOLD_DIMENSION = 15
global_step = tf.Variable(0)


class DNNReferenceBot:

    def __init__(self, numPlayers):
        self.samples = []
        self.numPlayers = numPlayers
        self.name = "DNN reference bot"

        self.checkpointPath = "dnnbot_reference/cp-{epoch:05d}.ckpt"
        self.checkpointDir = os.path.dirname(self.checkpointPath)
        self.checkpointCallback = tf.keras.callbacks.ModelCheckpoint(self.checkpointPath,
                                                                     save_weights_only=True,
                                                                     verbose=1)
        self.inputTensors = []
        self.chosenMoves = []
        self.totalLoss = 0
        self.totalMoveScoreLoss = 0
        self.totalScoreLoss = 0
        self.lossCount = 0
        self.iterationNumber = 0
        self.gamesPlayed = 0
        self.isTrained = False
        self.samplesX = []
        self.samplesY = []
        self.lastTrainingSize = 0
        self.trainLossResults = []
        self.createModel()
        #self.summaryWriter = tf.contrib.summary.create_file_writer('logs', flush_millis=10000)
        self.epoch = 1
        self.rating = 1000
        self.nextFullTrainingEpoch = 1000
        if True:
            latest = tf.train.latest_checkpoint(self.checkpointDir)
            self.model.load_weights(latest)
            self.isTrained = True
        else:
            self.fitModel()

    def createModel(self):
        historicalInputs = keras.Input(shape=(6, getStateTensorDimension(self.numPlayers),), name='historical')
        y = layers.LSTM(256)(historicalInputs)

        inputs = keras.Input(shape=([getMoveTensorDimension(self.numPlayers)]), name='state')
        x = layers.Dense(512, activation='relu', name='dense_1', kernel_regularizer=keras.regularizers.l2(l=0.01))(inputs)
        x = layers.Dropout(0.3)(x)
        #lstmOutput, stateH, stateC = layers.LSTM(64, return_state=True, name='lstm')(x)
        #lstmState = [stateH, stateC]
        x = layers.Concatenate()([y, x])
        x = layers.Dense(512, activation='relu', name='dense_2', kernel_regularizer=keras.regularizers.l2(l=0.01))(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(512, activation='relu', name='dense_3', kernel_regularizer=keras.regularizers.l2(l=0.01))(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(512, activation='relu', name='dense_4', kernel_regularizer=keras.regularizers.l2(l=0.01))(x)
        x = layers.Dropout(0.3)(x)
        outputScore = layers.Dense(1, activation='sigmoid', name='outputScore')(x)

        self.model = keras.Model(inputs=[inputs, historicalInputs], outputs=outputScore)
        self.model.summary()
        print(self.model.input)
        #self.optimizer = tf.keras.optimizers.RMSprop()
        self.optimizer = tf.keras.optimizers.Adam()

    def loss(self, x, y):
        y_ = self.model(x)
        return tf.keras.losses.MeanSquaredError()(y, y_)
        # return tf.losses.mean_squared_error(labels=y, predictions=y_)

    def grad(self, inputs, targets):
        with tf.GradientTape() as tape:
            loss_value = self.loss(inputs, targets)
        return loss_value, tape.gradient(loss_value, self.model.trainable_variables)

    def getFileName(self, ind):
        return ("data/%d.gz" % ind)

    def fitModel(self):
        self.startEpoch()
        ind = 1
        while True:
            fileName = self.getFileName(ind)
            samplesX = []
            samplesY = []
            try:
                # with open(fileName, 'rb') as f:
                [samplesX, samplesY] = compress_pickle.load(fileName)
                #i = 0
                #historyTensors = []
                # for line in f:
                #    tensor = self.readList(line)
                #    if i == 0:
                #        stateTensor = tensor
                #    else:
                #        historyTensors.append(np.asarray(tensor).astype('float32'))
                #    i += 1
                #    if i == 7:
                #        samplesX.append([stateTensor, historyTensors])
                #        historyTensors = []
                #        i = 0
                # if len(samplesX) > 0:
                #    samplesY = tensor
            except FileNotFoundError:
                break
            print("Training on %s" % fileName)
            self.fitModelForData(samplesX, samplesY)
            if ind % 10 == 0:
                print("Loss: {:.4f}".format(self.epochLossAvg.result()))
            ind += 1
        if ind > 1:
            self.endEpoch()

    def startEpoch(self):
        self.startTime = time.time()
        # with self.summaryWriter.as_default(), tf.contrib.summary.always_record_summaries():
        #    tf.contrib.summary.scalar("rating", self.rating, step=self.epoch)

        self.epochLossAvg = 0  # tfe.metrics.Mean()

    def endEpoch(self):
        # end epoch
        self.trainLossResults.append(self.epochLossAvg.result())

        print("Loss: {:.4f}".format(self.epochLossAvg.result()))

        self.isTrained = True
        self.epoch += 1
        endTime = time.time()
        print("Training took %.2f seconds" % (endTime - self.startTime))

    def fitModelForData(self, samplesX, samplesY):

        batchSize = 128

        #permutation = np.random.permutation(len(samplesY))
        inputTensors = []
        historyTensors = []
        for [inputTensor, historyTensor] in samplesX:
            inputTensors.append(inputTensor)
            historyTensors.append(historyTensor)
        #inputTensors = [inputTensors[i] for i in permutation]
        #historyTensors = [historyTensors[i] for i in permutation]
        #samplesY = [samplesY[i] for i in permutation]
        for i in range(0, len(samplesX), batchSize):
            x = np.stack(inputTensors[i:i + batchSize], axis=0)
            historyX = np.stack(historyTensors[i:i + batchSize], axis=0)
            y = tf.reshape(np.stack(samplesY[i:i + batchSize], axis=0), [-1, 1])
            #x = tf.reshape(x, [1,-1])
            #y = tf.reshape(y, [1,-1])

            # Optimize the model
            loss_value, grads = self.grad([x, historyX], y)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables), global_step)

            # Track progress
            self.epochLossAvg(loss_value)  # add current batch loss

    def getBestPayOption(self, payOptions):
        best = None
        for payOption in payOptions:
            if (best is None or payOption.totalCost() < bestSum):
                bestSum = payOption.totalCost()
                best = payOption
        return best

    def getPlayerStateTensor(self, state: State, player):
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

    def getHandTensor(self, state):
        tensor = np.zeros((len(ALL_CARDS)))
        index = 0
        for i in range(len(ALL_CARDS)):
            card = ALL_CARDS[i]
            if card in state.players[0].hand:
                tensor[index] = 1
            index += 1
        return tensor

    def getAgeAndPickTensor(self, state):
        tensor = np.zeros((9))
        tensor[state.age - 1] = 1
        tensor[10 - len(state.players[0].hand)] = 1
        return tensor

    def getStateTensor(self, state):
        tensor = np.concatenate((self.getAgeAndPickTensor(state), self.getHandTensor(state)))
        for player in range(state.numPlayers):
            tensor = np.concatenate((tensor, self.getPlayerStateTensor(state, player)))
        return tensor.astype('float32')

    def getMoveTensor(self, state, move):
        stateTensor = self.getStateTensor(state)
        tensor = np.zeros((len(ALL_CARDS) + PER_CARD_DIMENSION))
        if move.discard:
            tensor[0] = 1
        else:
            tensor[2 + move.payOption.totalCost()] = 1
        if move.buildWonder:
            tensor[1] = 1
        tensor[PER_CARD_DIMENSION + getCardIndex(move.card)] = 1
        return np.concatenate((stateTensor, tensor))

    def getMove(self, state):
        return self.getMoves([state])

    def getMoves(self, states):
        allMoves = []
        allMoveScores = []
        allTensors = []
        allHistoryTensors = []
        for stateInd in range(len(states)):
            state = states[stateInd]
            player = state.players[0]
            player.stateTensors.append(self.getStateTensor(state))
            playableCards = player.getPlayableCards()
            moves = []
            for card in player.hand:
                moves.append(Move(card=card, discard=True))
                if player.numWonderStagesBuilt < len(player.wonder.stages):
                    payOptions = state.getPayOptionsForCost(0, player.wonder.stages[player.numWonderStagesBuilt].cost)
                    for payOption in payOptions:
                        if (payOption.totalCost() <= player.gold):
                            moves.append(Move(card=card, buildWonder=True, wonderStageIndex=player.numWonderStagesBuilt, payOption=payOption))
            for card in playableCards:
                payOptions = state.getCardPayOptions(0, card)
                for payOption in payOptions:
                    if (payOption.totalCost() <= player.gold):
                        moves.append(Move(card=card, payOption=payOption))
            historyTensors = []
            while (len(historyTensors) + len(player.stateTensors) < 6):
                historyTensors.append(player.stateTensors[0])
            while (len(historyTensors) < 6):
                historyTensors.append(player.stateTensors[len(historyTensors) - 6])
            historyTensors = np.stack(historyTensors, axis=0)
            moveScores = []
            tensors = []
            myHistoryTensors = []
            for move in moves:
                state.players[0].performMove(move, removeCardFromHand=False)
                tensor = self.getMoveTensor(state, move)
                state.players[0].undoMove(move)
                tensors.append(tensor)
                myHistoryTensors.append(historyTensors)
            allMoves.append(moves)
            allMoveScores.append(moveScores)
            allTensors.append(tensors)
            allHistoryTensors.append(myHistoryTensors)
        flatTensors = [tensor for tensorList in allTensors for tensor in tensorList]
        flatHistoryTensors = [tensor for tensorList in allHistoryTensors for tensor in tensorList]
        scores = self.model([np.stack(flatTensors, axis=0), np.stack(flatHistoryTensors, axis=0).astype('float32')])
        scoreInd = 0
        chosenMoves = []
        for stateInd in range(len(states)):
            state = states[stateInd]
            moves = allMoves[stateInd]
            moveScores = allMoveScores[stateInd]
            tensors = allTensors[stateInd]
            historyTensors = allHistoryTensors[stateInd]
            player = state.players[0]
            for i in range(len(moves)):
                if self.isTrained:
                    score = scores[scoreInd][0]
                    scoreInd += 1
                else:
                    score = 0.5
                moveScores.append(MoveScore(moves[i], score, [tensors[i], historyTensors[i]]))
            moveScores.sort(key=lambda x: -x.score)
            bestMoveScore = moveScores[0].score
            totalPriority = 0
            #n = self.gamesPlayed+2
            n = 1000000
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
                print('Reference bot\'s scores:')
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

    def train(self, state):
        return
        scores = []
        for i in range(state.numPlayers):
            scores.append(state.getScore(i))
        wonGame = True
        actualScore = 0
        for i in range(1, state.numPlayers):
            scoreDiff = scores[0] - scores[i]
            if scoreDiff == 0:
                value = 0.5
            else:
                if scoreDiff > 0:
                    scoreDiff += 5
                else:
                    scoreDiff -= 5
                value = 0.5 + 0.5 * np.tanh(scoreDiff * 0.05)
            if scores[i] >= scores[0]:
                wonGame = False
            actualScore += value / (state.numPlayers - 1)
        winValue = 0.4
        actualScore *= (1 - winValue)
        if wonGame:
            actualScore += winValue
        actualScore = 0.7 * actualScore + 0.3 * (0.5 + 0.5 * np.tanh((scores[0] - 50) * 0.05))
        for i in range(len(state.players[0].tensors)):
            tensor = state.players[0].tensors[i]
            self.samplesX.append(tensor)
            if i == len(state.players[0].tensors) - 1:
                self.samplesY.append(actualScore)
            else:
                self.samplesY.append(state.players[0].bestMoveScores[i + 1])
        self.gamesPlayed += 1
        samplesWhenBad = 540
        samplesWhenGood = 5400
        if self.rating < 1400:
            samplesNeeded = samplesWhenBad
        else:
            samplesNeeded = samplesWhenGood
        if len(self.samplesY) > samplesNeeded:
            self.lastTrainingSize = len(self.samplesY)
            ind = 1
            if (self.rating > 1400):
                while True:
                    fileName = self.getFileName(ind)
                    if not os.path.isfile(fileName):
                        # with open(fileName, 'wb') as f:
                        compress_pickle.dump([self.samplesX, self.samplesY], fileName)
                        # for [stateTensor, historyTensors] in self.samplesX:
                        #    self.writeList(f, stateTensor)
                        #    assert(len(historyTensors) == 6)
                        #    for historyTensor in historyTensors:
                        #        self.writeList(f, historyTensor)
                        #self.writeList(f, self.samplesY)
                        break
                    ind += 1
                    maxNumFiles = 10000
                    if (ind > maxNumFiles):
                        ind = random.randint(1, maxNumFiles)
                        fileName = self.getFileName(ind)
                        os.remove(fileName)
                        ind = 1
            self.startEpoch()
            self.fitModelForData(self.samplesX, self.samplesY)
            self.endEpoch()
            self.samplesX = []
            self.samplesY = []
            self.model.save_weights(self.checkpointPath.format(epoch=self.epoch))
            if self.epoch >= self.nextFullTrainingEpoch:
                print('Training on all data')
                self.fitModel()
                self.nextFullTrainingEpoch *= 1.3
                self.nextFullTrainingEpoch = max(self.nextFullTrainingEpoch * 1.3, self.epoch + 1)

    def writeList(self, f, tensor):
        for value in tensor:
            f.write("%f " % value)
        f.write("\n")

    def readList(self, line):
        return [float(x) for x in line.split()]
