import math
from elo import rate_1vs1
from control import playGames
from random_bot import RandomBot
from science_bot import ScienceBot
# from dnn_reference_bot import DNNReferenceBot
from pytorch_bot import TorchBot

randomBot = RandomBot()
scienceBot = ScienceBot()
# dnnReferenceBot = dnnReferenceBot(3)
# dnnReferenceBot = DNNReferenceBot(3)
torchBot = TorchBot(3)

# random.seed(2)
bots = [randomBot, scienceBot, torchBot]
debug = True
gamesAtATime = 1 if debug else 100
for bot in bots:
    bot.rating = 1000
games = 0
while True:
    print('Game %d' % (games + 1))
    if ((games // gamesAtATime) % 10 == 0):
        testingMode = True
    else:
        testingMode = False
    if debug:
        testingMode = True
    if testingMode:
        bots = [torchBot, torchBot, torchBot]
    else:
        bots = [torchBot, torchBot, torchBot]
    for bot in bots:
        bot.testingMode = testingMode
    playGames(bots, gamesAtATime)
    factor = min(5, 50.0 / math.pow(games + 1, 0.8))
    if testingMode:
        for gameInd in range(gamesAtATime):
            for i in range(len(bots)):
                for j in range(i):
                    if bots[i] == bots[j]:
                        continue
                    if bots[i].scores[gameInd] > bots[j].scores[gameInd]:
                        # print('%s defeats %s' % (bots[i].name, bots[j].name))
                        (newRatingI, newRatingJ) = rate_1vs1(bots[i].rating, bots[j].rating)
                    elif bots[i].scores[gameInd] == bots[j].scores[gameInd]:
                        # print('%s and %s draw' % (bots[i].name, bots[j].name))
                        (newRatingI, newRatingJ) = rate_1vs1(bots[i].rating, bots[j].rating, drawn=True)
                    else:
                        # print('%s defeats %s' % (bots[j].name, bots[i].name))
                        (newRatingJ, newRatingI) = rate_1vs1(bots[j].rating, bots[i].rating)
                    bots[i].rating = bots[i].rating * (1 - factor) + newRatingI * factor
                    bots[j].rating = bots[j].rating * (1 - factor) + newRatingJ * factor
                    randomBot.rating = 1000
                    dnnReferenceBot.rating = 1720
        for bot in bots:
            print('%s\'s rating: %d' % (bot.name, bot.rating))
    games += gamesAtATime
