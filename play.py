import math
from elo import rate_1vs1
from control import playGames
from random_bot import RandomBot
from science_bot import ScienceBot
# from dnn_reference_bot import DNNReferenceBot
from pytorch_bot import TorchBot
import numpy as np

randomBot = RandomBot()
scienceBot = ScienceBot()
# dnnReferenceBot = DNNReferenceBot(3)
# dnnReferenceBot = DNNReferenceBot(3)
torchBot = TorchBot(3)

# random.seed(2)
bots = [randomBot, scienceBot, torchBot]
debug = False
gamesAtATime = 1 if debug else 100
for bot in bots:
    bot.rating = 1000
games = 0
all_scores = None
while True:
    print('Game %d' % (games + 1))
    if ((games // gamesAtATime) % 10 == 0):
        testingMode = True
    else:
        testingMode = False
    if debug:
        testingMode = True
    if testingMode:
        bots = [torchBot, randomBot, randomBot]
    else:
        bots = [torchBot, randomBot, randomBot]
    for bot in bots:
        bot.testingMode = testingMode
    scores = playGames(bots, gamesAtATime)
    all_scores = np.concatenate([all_scores, scores]) if all_scores is not None else scores
    factor = min(5, 50.0 / math.pow(games + 1, 0.8))
    if testingMode:
        for gameInd in range(scores.shape[0]):
            for i in range(len(bots)):
                for j in range(i):
                    if bots[i] == bots[j]:
                        continue
                    if scores[gameInd, i] > scores[gameInd, j]:
                        # print('%s defeats %s' % (bots[i].name, bots[j].name))
                        (newRatingI, newRatingJ) = rate_1vs1(bots[i].rating, bots[j].rating)
                    elif scores[gameInd, i] == scores[gameInd, j]:
                        # print('%s and %s draw' % (bots[i].name, bots[j].name))
                        (newRatingI, newRatingJ) = rate_1vs1(bots[i].rating, bots[j].rating, drawn=True)
                    else:
                        # print('%s defeats %s' % (bots[j].name, bots[i].name))
                        (newRatingJ, newRatingI) = rate_1vs1(bots[j].rating, bots[i].rating)
                    bots[i].rating = bots[i].rating * (1 - factor) + newRatingI * factor
                    bots[j].rating = bots[j].rating * (1 - factor) + newRatingJ * factor
                    randomBot.rating = 1000
                    # dnnReferenceBot.rating = 1720
        for bot in bots:
            print('%s\'s rating: %d' % (bot.name, bot.rating))
    games += gamesAtATime
