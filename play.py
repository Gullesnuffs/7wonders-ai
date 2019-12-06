import math
import random
from card import Color
from elo import rate_1vs1
from control import playGames
from random_bot import RandomBot
from science_bot import ScienceBot
# from dnn_reference_bot import DNNReferenceBot
from pytorch_bot import TorchBot, CardBonuses
from pytorch_bot2 import ParameterizedTorchBot
from pytorch_reference_bot import TorchReferenceBot
import numpy as np

randomBot = RandomBot()
scienceBot = ScienceBot()
numPlayers = 5
# dnnReferenceBot = DNNReferenceBot(3)
# dnnReferenceBot = DNNReferenceBot(3)
torchBot = ParameterizedTorchBot(numPlayers, 'pytorchbot/parameterizedtorchbot.pt', 'ParameterizedTorchBot')
scienceCardBonuses = CardBonuses()
scienceCardBonuses.set_color_bonus(Color.GREEN, 0.003)
scienceTorchBot = TorchBot(numPlayers, 'pytorchbot/scienceTorchBot.pt', scienceCardBonuses, 'ScienceTorchBot')
militaryCardBonuses = CardBonuses()
militaryCardBonuses.set_color_bonus(Color.RED, 0.003)
militaryTorchBot = TorchBot(numPlayers, 'pytorchbot/militaryTorchBot.pt', militaryCardBonuses, 'MilitaryTorchBot')
civilianCardBonuses = CardBonuses()
civilianCardBonuses.set_color_bonus(Color.BLUE, 0.003)
civilianTorchBot = TorchBot(numPlayers, 'pytorchbot/civilianTorchBot.pt', civilianCardBonuses, 'CivilianTorchBot')
torchBots = [torchBot, scienceTorchBot, militaryTorchBot, civilianTorchBot]
torchReferenceBot = TorchReferenceBot(numPlayers)

# random.seed(2)
debug = False
gamesAtATime = 1 if debug else 100
games = 0
all_scores = None
while True:
    torchBot.nash.printState()
    print('Game %d' % (games + 1))
    if ((games // gamesAtATime) % 10 == 0):
        testingMode = True
    else:
        testingMode = False
    if debug:
        testingMode = True
    if testingMode:
        bots = [torchBot, scienceTorchBot, randomBot, militaryTorchBot, civilianTorchBot]
    else:
        bots = [torchBot, torchBot, torchBot, torchBot, torchBot]
        #for i in range(numPlayers):
        #    bots.append(random.choice(torchBots))
    for bot in bots:
        bot.testingMode = testingMode
    scores = playGames(bots, gamesAtATime)
    all_scores = np.concatenate([all_scores, scores]) if all_scores is not None else scores
    if testingMode:
        for i in range(len(bots)):
            bots[i].rating = 1000
        volatility = 20
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
            bot.onRatingsAssigned()
    games += gamesAtATime
