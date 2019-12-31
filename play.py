import math
import random
from card import Color
from elo import rate_1vs1
from control import playGames
from random_bot import RandomBot
from science_bot import ScienceBot
# from dnn_reference_bot import DNNReferenceBot
from pytorch_bot import TorchBot
from pytorch_reference_bot import TorchReferenceBot
from rollout_bot import RolloutBot
import numpy as np

randomBot = RandomBot()
scienceBot = ScienceBot()
numPlayers = 6
# dnnReferenceBot = DNNReferenceBot(3)
# dnnReferenceBot = DNNReferenceBot(3)
torchBot = TorchBot(numPlayers, 'pytorchbot/torchbot.pt', 'TorchBot')
rolloutBot = RolloutBot(numPlayers, 'pytorchbot/torchbot.pt', 'RolloutBot')

#random.seed(3)
debug = False
games = 0
all_scores = None
iterationNumber = 0
testRolloutBot = True
while True:
    torchBot.nash.printState()
    print('Game %d' % (games + 1))
    if (iterationNumber % 10 == 0 or testRolloutBot):
        testingMode = True
    else:
        testingMode = False
    #testingMode = True
    iterationNumber += 1
    if debug:
        testingMode = True
    bots = []
    for i in range(numPlayers):
        bots.append(torchBot)
    if testingMode:
        bots[-1] = randomBot
        if testRolloutBot:
            bots[-1] = rolloutBot
    for bot in bots:
        bot.testingMode = testingMode
    gamesAtATime = 1 if (debug or (testingMode and testRolloutBot)) else 100
    scores = playGames(bots, gamesAtATime)
    all_scores = np.concatenate([all_scores, scores]) if all_scores is not None else scores
    if testingMode or True:
        for i in range(len(bots)):
            bots[i].rating = 1000
        volatility = 0.3
        while volatility > 0.001:
            simpleScore = [0 for bot in bots]
            for gameInd in range(scores.shape[0]):
                for i in range(len(bots)):
                    for j in range(i):
                        if bots[i] == bots[j]:
                            continue
                        if scores[gameInd, i] > scores[gameInd, j]:
                            (newRatingI, newRatingJ) = rate_1vs1(bots[i].rating, bots[j].rating)
                            simpleScore[i] += 1
                        elif scores[gameInd, i] == scores[gameInd, j]:
                            (newRatingI, newRatingJ) = rate_1vs1(bots[i].rating, bots[j].rating, drawn=True)
                            simpleScore[i] += 0.5
                            simpleScore[j] += 0.5
                        else:
                            (newRatingJ, newRatingI) = rate_1vs1(bots[j].rating, bots[i].rating)
                            simpleScore[j] += 1
                        bots[i].rating = bots[i].rating * (1 - volatility) + newRatingI * volatility
                        bots[j].rating = bots[j].rating * (1 - volatility) + newRatingJ * volatility
                        randomBot.rating = 1000
                        # torchReferenceBot.rating = 1650
                        # dnnReferenceBot.rating = 1720
            volatility /= 1.5
        for i in range(len(bots)):
            print("%s: %.1f" % (bots[i].name, simpleScore[i]))
        for bot in bots:
            print('%s\'s rating: %d' % (bot.name, bot.rating))
            if testingMode:
                bot.onRatingsAssigned()
    games += gamesAtATime
